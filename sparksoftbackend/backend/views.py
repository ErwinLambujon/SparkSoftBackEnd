from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
import logging
from django.core.files.storage import default_storage
from django.conf import settings

from langchain_aws import BedrockLLM
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator

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
    indexes_with_filenames = []

    for pdf_file in pdf_files:
        if pdf_file.name.lower().endswith(".pdf"):
            try:
                logger.info(f"Processing {pdf_file.name}...")
                file_path = default_storage.save(pdf_file.name, pdf_file)
                full_path = os.path.join(settings.MEDIA_ROOT, file_path)

                data_load = PyPDFLoader(full_path)
                data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=300,
                                                            chunk_overlap=10)

                loaded_docs = data_load.load()
                docs = []
                for doc in loaded_docs:
                    chunks = data_split.split_text(doc.page_content)
                    docs.extend(chunks)

                data_embeddings = BedrockEmbeddings(credentials_profile_name='default',
                                                    model_id='amazon.titan-embed-text-v2:0')
                data_index = VectorstoreIndexCreator(text_splitter=data_split, embedding=data_embeddings,
                                                     vectorstore_cls=FAISS)
                index = data_index.from_loaders([data_load])
                indexes_with_filenames.append((index, file_path))

                logger.info(f"Processed successfully {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        else:
            logger.warning(f"File {pdf_file.name} is not a PDF. Skipping this file.")

    if not indexes_with_filenames:
        raise ValueError("No documents were loaded and processed.")

    return indexes_with_filenames

def hr_rag_response(indexes_with_filenames, question):
    rag_llm = hr_llm()
    responses = []
    for index, file_path in indexes_with_filenames:
        try:
            response = index.query(question=question, llm=rag_llm)
            file_name = os.path.basename(file_path)
            responses.append((response, file_name))
        except Exception as e:
            logger.error(f"Error querying index for file {file_path}: {e}")
            file_name = os.path.basename(file_path)
            responses.append((None, file_name))
    return responses

@api_view(['POST'])
def upload_files(request):
    logger.info(f"Received upload request. Files: {[f.name for f in request.FILES.getlist('files')]}")
    try:
        files = request.FILES.getlist('files')
        if not files:
            logger.warning("No files provided in the request")
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        indexes_with_filenames = hr_index(files)
        logger.info(f"Indexes created: {indexes_with_filenames}")

        return Response({'message': 'Files processed successfully'}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}", exc_info=True)
        return Response({'error': f'An error occurred while processing the files: {str(e)}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['POST'])
def ask_ai(request):
    logger.info(f"Received request data: {request.data}")
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

