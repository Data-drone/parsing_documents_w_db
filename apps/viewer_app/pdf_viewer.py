import streamlit as st
from databricks.sdk import WorkspaceClient
import tempfile
import os
import base64
import logging
from streamlit_pdf_viewer import pdf_viewer

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get the full volume path from the environment variable
VOLUME_PATH = os.environ.get("DATABRICKS_VOLUME", "")

st.set_page_config(page_title="Databricks Volume PDF Viewer", layout="wide")
st.title("Databricks Volume PDF Viewer (SDK)")
st.markdown(f"**Current Volume Path:** `{VOLUME_PATH}` (set via $DATABRICKS_VOLUME env var)")
st.markdown("""
This app lets you browse and preview PDF files stored in a Databricks Unity Catalog volume.\
- Use the sidebar to refresh and select a PDF file.\
- The selected PDF will be displayed below with page navigation, zoom, search, and download tools.
""")

# List files in the given volume path
def list_files_in_volume():
    w = WorkspaceClient()
    try:
        files = [f.path for f in w.files.list_directory_contents(VOLUME_PATH)]
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []

# Download a PDF file and return a local path for Streamlit to preview
def download_pdf_file(file_path):
    w = WorkspaceClient()
    try:
        local_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        try:
            import IPython
            dbutils = IPython.get_ipython().user_ns.get('dbutils')
            if dbutils:
                dbutils.fs.cp(file_path, f"file:{local_temp_path}")
                return local_temp_path
        except:
            pass
        response = w.files.download(file_path)
        with open(local_temp_path, 'wb') as f:
            if hasattr(response, 'contents'):
                content = response.contents
                if hasattr(content, 'read'):
                    data = content.read()
                    f.write(data if isinstance(data, bytes) else data.encode())
                elif hasattr(content, '__iter__'):
                    for chunk in content:
                        f.write(chunk if isinstance(chunk, bytes) else str(chunk).encode())
                else:
                    f.write(bytes(content))
            elif hasattr(response, 'read'):
                f.write(response.read())
            elif hasattr(response, 'data'):
                f.write(response.data if isinstance(response.data, bytes) else response.data.encode())
            else:
                f.write(bytes(response))
        return local_temp_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        import traceback
        traceback.print_exc()
        return None

# Sidebar for file selection
with st.sidebar:
    st.header("PDF File Browser")
    if st.button("🔄 Refresh PDF List") or 'pdf_files' not in st.session_state:
        logger.info("Refreshing the file list from volume: %s", VOLUME_PATH)
        st.session_state['pdf_files'] = list_files_in_volume()
    pdf_files = st.session_state.get('pdf_files', [])
    selected_file = st.selectbox("Select a PDF file", options=pdf_files, format_func=lambda x: os.path.basename(x) if x else "")

# Main area for PDF display
if selected_file:
    logger.info("Loading PDF: %s", selected_file)
    local_path = download_pdf_file(selected_file)
    if local_path:
        logger.info("PDF Loaded: %s", selected_file)
        pdf_viewer(local_path, width=900, height=900)
        # Clean up temp file after displaying
        try:
            os.unlink(local_path)
        except:
            pass
    else:
        st.error("Error loading PDF.")
else:
    st.info("Select a PDF file to preview.") 