from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
import logging
from django.core.files.storage import default_storage
from django.conf import settings

from langchain_aws import BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Add this line

from PyPDF2 import PdfReader
import io

logger = logging.getLogger(__name__)

# Global variable to store indexes
indexes_with_filenames = []

def hr_llm():
    llm = BedrockLLM(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.5
        }
    )
    return llm


def hr_index(pdf_files):
    global indexes_with_filenames
    new_indexes = []

    for pdf_file in pdf_files:
        if pdf_file.name.lower().endswith(".pdf"):
            try:
                logger.info(f"Processing {pdf_file.name}...")
                file_content = pdf_file.read()
                pdf_reader = PdfReader(io.BytesIO(file_content))

                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()

                if not text_content.strip():
                    logger.warning(f"No text content extracted from {pdf_file.name}")
                    continue

                data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=300,
                                                            chunk_overlap=10)
                chunks = data_split.split_text(text_content)

                documents = [Document(page_content=chunk, metadata={"source": pdf_file.name}) for chunk in chunks]

                data_embeddings = BedrockEmbeddings(credentials_profile_name='default',
                                                    model_id='amazon.titan-embed-text-v2:0')
                vectorstore = FAISS.from_documents(documents, data_embeddings)

                file_path = default_storage.save(pdf_file.name, pdf_file)
                new_indexes.append((vectorstore, file_path))

                logger.info(f"Processed successfully {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"File {pdf_file.name} is not a PDF. Skipping this file.")

    if not new_indexes:
        logger.warning("No documents were successfully loaded and processed.")
        return []

    indexes_with_filenames.extend(new_indexes)
    logger.info(f"Total indexes with filenames: {indexes_with_filenames}")
    return new_indexes


def hr_rag_response(indexes_with_filenames, question):
    rag_llm = hr_llm()
    responses = []
    for vectorstore, file_path in indexes_with_filenames:
        try:
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Context: {context}

            Question: {question}

            Please answer the question based on the given context. If the answer cannot be found in the context, please say so.
            """

            response = rag_llm(prompt)
            file_name = os.path.basename(file_path)
            responses.append((response, file_name))
        except Exception as e:
            logger.error(f"Error querying index for file {file_path}: {e}")
            file_name = os.path.basename(file_path)
            responses.append((None, file_name))
    return responses

@api_view(['POST'])
def upload_files(request):
    global indexes_with_filenames
    logger.info(f"Received upload request. Files: {[f.name for f in request.FILES.getlist('files')]}")
    try:
        files = request.FILES.getlist('files')
        if not files:
            logger.warning("No files provided in the request")
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        new_indexes = hr_index(files)
        logger.info(f"New indexes created: {new_indexes}")

        if not new_indexes:
            return Response({
                'message': 'No files were successfully processed',
                'processed_files': []
            }, status=status.HTTP_200_OK)

        return Response({
            'message': 'Files processed successfully',
            'processed_files': [os.path.basename(file_path) for _, file_path in new_indexes]
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}", exc_info=True)
        return Response({'error': f'An error occurred while processing the files: {str(e)}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def ask_ai(request):
    global indexes_with_filenames
    logger.info(f"Received request data: {request.data}")
    logger.info(f"Current indexes_with_filenames: {indexes_with_filenames}")
    try:
        question = request.data.get('question')
        if not question:
            logger.warning("No question provided in the request")
            return Response({'error': 'No question provided'}, status=status.HTTP_400_BAD_REQUEST)

        if not indexes_with_filenames:
            logger.warning("No documents have been processed yet")
            return Response({'error': 'No documents have been processed yet'}, status=status.HTTP_400_BAD_REQUEST)

        responses = hr_rag_response(indexes_with_filenames, question)
        logger.info(f"Generated responses: {responses}")
        return Response({'responses': responses})
    except Exception as e:
        logger.error(f"Error in ask_ai: {e}", exc_info=True)
        return Response({'error': 'An error occurred while processing the request'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
