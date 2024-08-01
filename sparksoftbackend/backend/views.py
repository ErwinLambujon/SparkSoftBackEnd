from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import logging
from django.core.files.storage import default_storage
from langchain.chains import RetrievalQA
from langchain_aws import BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

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

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(pdf_file.read())
                    temp_file_path = temp_file.name

                # Use PyPDFLoader with the temporary file
                data_load = PyPDFLoader(temp_file_path)
                documents = data_load.load()

                # Split the documents
                text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=300,
                                                               chunk_overlap=10)
                splits = text_splitter.split_documents(documents)

                # Create embeddings and vector store
                embeddings = BedrockEmbeddings(credentials_profile_name='default',
                                               model_id='amazon.titan-embed-text-v2:0')
                vectorstore = FAISS.from_documents(splits, embeddings)

                # Save the file and add to indexes
                file_path = default_storage.save(pdf_file.name, pdf_file)
                new_indexes.append((vectorstore, file_path))

                logger.info(f"Processed successfully {pdf_file.name}")

                # Clean up the temporary file
                os.unlink(temp_file_path)

            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"File {pdf_file.name} is not a PDF. Skipping this file.")

    if not new_indexes:
        logger.warning("No documents were successfully loaded and processed.")
        return []

    indexes_with_filenames.extend(new_indexes)
    logger.info(f"Total indexes with filenames: {len(indexes_with_filenames)}")
    return new_indexes


def hr_rag_response(indexes_with_filenames, question, context):
    rag_llm = hr_llm()
    all_results = []

    for vectorstore, file_path in indexes_with_filenames:
        try:
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=rag_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            file_name = os.path.basename(file_path)
            all_results.append((result['result'], file_name))
        except Exception as e:
            logger.error(f"Error querying index for file {file_path}: {e}", exc_info=True)
            file_name = os.path.basename(file_path)
            all_results.append((None, file_name))

    # Combine all results
    combined_context = "\n\n".join([f"From {file_name}:\n{result}" for result, file_name in all_results if result])

    # Append the previous conversation context
    conversation_context = "\n\n".join([f"User: {entry['user']}\nAI: {entry.get('ai', '')}" for entry in context])

    # Create a new prompt for the final answer
    prompt_engineering = """
    Based on the following information from multiple documents:

    {context}

    And the previous conversation context:

    {conversation_context}

    Please provide a comprehensive answer to the question: {question}
    If the information is not available or if there are contradictions, please mention that.
    """

    prompt = PromptTemplate(template=prompt_engineering, input_variables=["context", "conversation_context", "question"])

    # Create a new chain for the final answer
    final_chain = LLMChain(llm=rag_llm, prompt=prompt)

    # Generate the final answer
    final_answer = final_chain.run(context=combined_context, conversation_context=conversation_context, question=question)

    return [("Combined answer from all documents:\n\n" + final_answer, "All Documents")]

@api_view(['POST'])
def upload_files(request):
    global indexes_with_filenames
    logger.info(f"Received upload request. Files: {request.FILES}")
    try:
        files = request.FILES.getlist('files')
        if not files:
            logger.warning("No files provided in the request")
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        new_indexes = hr_index(files)
        logger.info(f"New indexes created: {new_indexes}")

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
        context = request.data.get('context', [])
        if not question:
            logger.warning("No question provided in the request")
            return Response({'error': 'No question provided'}, status=status.HTTP_400_BAD_REQUEST)

        if not indexes_with_filenames:
            logger.warning("No documents have been processed yet")
            return Response({'error': 'No documents have been processed yet'}, status=status.HTTP_400_BAD_REQUEST)

        responses = hr_rag_response(indexes_with_filenames, question, context)
        logger.info(f"Generated responses: {responses}")
        return Response({'responses': responses})
    except Exception as e:
        logger.error(f"Error in ask_ai: {e}", exc_info=True)
        return Response({'error': 'An error occurred while processing the request'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
