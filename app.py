import os
import logging
import tempfile
import streamlit as st
from llama_parse import LlamaParse

# Configure logging
logging.basicConfig(level=logging.INFO)

class LlamaParseAgentService:
    """
    Service for parsing documents into structured markdown tables using LlamaParse using a custom prompt formatting.
    """

    def __init__(self, prompt: str, api_key: str, model: str = "parse_document_with_agent"):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not provided")
        self.prompt = prompt
        self.model = model 

    def _collate_markdown_output(self, parsed_markdown):
        """
        Collect all markdown content from parsed results.
        """
        collated_markdown = "\n\n".join([doc.text for doc in parsed_markdown])
        return collated_markdown

    def parse_document(self, file_path: str) -> str:
        """
        Parse a document to markdown tables using LlamaParse and the loaded prompt.
        """
        try:
            parser = LlamaParse(
                result_type="markdown",
                parse_mode=self.model,
                complemental_formatting_instruction=self.prompt,
                user_prompt=self.prompt,
                api_key=self.api_key
            )
            document = parser.load_data(file_path)
            logging.info(f"✓ Successfully parsed document: {file_path}")
            return self._collate_markdown_output(document)
        
        except Exception as e:
            logging.error(f"✗ Error parsing document {file_path}: {e}")
            raise

# Streamlit App
def main():
    st.set_page_config(
        page_title="PDF to English Translator",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Translation to English")
    st.markdown("Upload a PDF document in any language and translate it to English markdown.")
    
    # Get API key from environment variable
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        if api_key:
            st.success("✓ API Key configured")
        else:
            st.error("⚠️ API Key not found in environment variables")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This app uses LlamaParse to extract and translate PDF content "
            "to English markdown format."
        )
        st.markdown(
            "Get your API key at [cloud.llamaindex.ai](https://cloud.llamaindex.ai)"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document in any language"
        )
        
        if uploaded_file is not None:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size / 1024:.2f} KB")
    
    with col2:
        st.header("📝 Translation Output")
        
    # Process button
    if uploaded_file is not None:
        if st.button("🚀 Translate to English", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ LLAMA_CLOUD_API_KEY environment variable not set. Please configure it in your deployment settings.")
            else:
                try:
                    with st.spinner("Processing document... This may take a moment."):
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Translation prompt
                        translation_prompt = """
                        Translate all text in this document to English. 
                        Preserve the original formatting, structure, and layout as markdown.
                        If the text is already in English, keep it as is.
                        Maintain all tables, lists, headers, and formatting elements.
                        """
                        
                        # Initialize service and parse
                        service = LlamaParseAgentService(
                            prompt=translation_prompt,
                            api_key=api_key
                        )
                        
                        markdown_output = service.parse_document(tmp_file_path)
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        # Display results
                        st.success("✓ Translation completed!")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["📖 Rendered Markdown", "📋 Raw Markdown"])
                        
                        with tab1:
                            st.markdown(markdown_output)
                        
                        with tab2:
                            st.code(markdown_output, language="markdown")
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Markdown",
                            data=markdown_output,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_translated.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    logging.error(f"Error details: {e}", exc_info=True)
    
    # Instructions
    if uploaded_file is None:
        st.info(
            "👆 Please upload a PDF file to begin translation. "
            "The document can be in any language and will be translated to English."
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by LlamaParse | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()