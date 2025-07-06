import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks import sql
import tempfile
import os
import base64
import logging
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
VOLUME_PATH = os.environ.get("DATABRICKS_VOLUME", "")
SQL_WAREHOUSE = os.environ.get("DATABRICKS_SQL_WAREHOUSE", "")

# Databricks configuration
cfg = Config()

st.set_page_config(page_title="Databricks Volume PDF Viewer", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Select Page",
        ["View Volumes", "View Markdown Extraction", "View VLM Extraction"],
        index=0
    )
    st.divider()

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

# Cache the SQL connection - following official Databricks app template pattern
@st.cache_resource
def get_connection(warehouse):
    """Create a cached connection to Databricks SQL warehouse"""
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse}",
        credentials_provider=lambda: cfg.authenticate,
    )

# Query the document_markdown table (optimized for list view)
def get_document_markdown_data():
    """Query the document_markdown table and return as a DataFrame (excluding large columns)"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = """
            SELECT 
                file_name,
                volume_path,
                file_extension,
                file_size_bytes,
                modification_time,
                directory,
                extraction_success,
                error_message,
                processing_time_seconds,
                character_count,
                page_count,
                processing_timestamp
            FROM brian_gen_ai.parsing_test.document_markdown
            ORDER BY processing_timestamp DESC
            """
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying document_markdown table: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Query full details for a specific document (only what we need for display)
def get_document_details(file_name):
    """Get full details for a specific document including markdown content"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = """
            SELECT 
                file_name,
                volume_path,
                directory,
                file_extension,
                extraction_success,
                page_count,
                character_count,
                processing_time_seconds,
                error_message,
                markdown_content
            FROM brian_gen_ai.parsing_test.document_markdown
            WHERE file_name = ?
            """
            cursor.execute(query, (file_name,))
            result = cursor.fetchall_arrow().to_pandas()
            return result.iloc[0] if len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error querying document details: {e}")
        return None

# VLM Extraction Functions
def get_vlm_document_list():
    """Get distinct source filenames from the document_store_ocr table"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = """
            SELECT DISTINCT 
                source_filename,
                MAX(total_pages) as total_pages,
                MAX(file_size_bytes) as file_size_bytes,
                MAX(processing_timestamp) as latest_processing
            FROM brian_gen_ai.parsing_test.document_store_ocr
            WHERE source_filename IS NOT NULL
            GROUP BY source_filename
            ORDER BY latest_processing DESC
            """
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying VLM document list: {e}")
        return pd.DataFrame()

def get_vlm_document_pages(source_filename):
    """Get all pages for a specific document from the document_store_ocr table"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = """
            SELECT 
                doc_id,
                source_filename,
                page_number,
                page_image_png,
                total_pages,
                file_size_bytes,
                processing_timestamp,
                metadata_json,
                ocr_text,
                ocr_timestamp,
                ocr_model
            FROM brian_gen_ai.parsing_test.document_store_ocr
            WHERE source_filename = ?
            ORDER BY page_number ASC
            """
            cursor.execute(query, (source_filename,))
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying VLM document pages: {e}")
        return pd.DataFrame()

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

# Page 1: View Volumes
if page == "View Volumes":
    st.title("Databricks Volume PDF Viewer (SDK)")
    st.markdown(f"**Current Volume Path:** `{VOLUME_PATH}` (set via $DATABRICKS_VOLUME env var)")
    st.markdown("""
    This app lets you browse and preview PDF files stored in a Databricks Unity Catalog volume.
    - Use the sidebar to refresh and select a PDF file.
    - The selected PDF will be displayed below with page navigation, zoom, search, and download tools.
    """)

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

# Page 2: View Markdown Extraction
elif page == "View Markdown Extraction":
    st.title("Markdown Extraction")
    st.markdown("**Extract and view markdown content from PDF files**")
    st.markdown(f"**SQL Warehouse:** `{SQL_WAREHOUSE}` (set via $DATABRICKS_SQL_WAREHOUSE env var)")
    
    # Add refresh button for the table data
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Data") or 'markdown_data' not in st.session_state:
            logger.info("Refreshing document markdown data from SQL warehouse")
            st.session_state['markdown_data'] = get_document_markdown_data()
    
    # Get the data
    df = st.session_state.get('markdown_data', pd.DataFrame())
    
    if not df.empty:
        st.subheader("Document Processing Results")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            successful_extractions = df['extraction_success'].sum() if 'extraction_success' in df.columns else 0
            st.metric("Successful Extractions", successful_extractions)
        with col3:
            total_pages = df['page_count'].sum() if 'page_count' in df.columns else 0
            st.metric("Total Pages", int(total_pages) if pd.notna(total_pages) else 0)
        with col4:
            avg_processing_time = df['processing_time_seconds'].astype(str).astype(float).mean() if 'processing_time_seconds' in df.columns else 0
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s" if pd.notna(avg_processing_time) else "N/A")
        
        # Filter options
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        with col1:
            if 'extraction_success' in df.columns:
                success_filter = st.selectbox(
                    "Filter by Extraction Status",
                    options=['All', 'Successful', 'Failed'],
                    index=0
                )
        with col2:
            if 'file_extension' in df.columns:
                extensions = ['All'] + sorted(df['file_extension'].dropna().unique().tolist())
                extension_filter = st.selectbox(
                    "Filter by File Extension", 
                    options=extensions,
                    index=0
                )
        
        # Apply filters
        filtered_df = df.copy()
        if 'extraction_success' in df.columns and success_filter != 'All':
            if success_filter == 'Successful':
                filtered_df = filtered_df[filtered_df['extraction_success'] == True]
            elif success_filter == 'Failed':
                filtered_df = filtered_df[filtered_df['extraction_success'] == False]
        
        if 'file_extension' in df.columns and extension_filter != 'All':
            filtered_df = filtered_df[filtered_df['file_extension'] == extension_filter]
        
        # Display the main table
        st.subheader("Document Details")
        
        # Select only filename and page_count for the main display
        display_columns = []
        if 'file_name' in filtered_df.columns:
            display_columns.append('file_name')
        if 'page_count' in filtered_df.columns:
            display_columns.append('page_count')
        
        # Show the simplified table
        if display_columns:
            display_df = filtered_df[display_columns].copy()
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "file_name": "File Name",
                    "page_count": "Page Count"
                }
            )
            
            # Show detailed view for selected row
            if len(display_df) > 0:
                st.subheader("Row Details")
                selected_idx = st.selectbox(
                    "Select a document to view details:",
                    options=range(len(display_df)),
                    format_func=lambda x: display_df.iloc[x]['file_name'] if 'file_name' in display_df.columns else f"Row {x+1}"
                )
                
                if selected_idx is not None:
                    selected_file_name = filtered_df.iloc[selected_idx]['file_name']
                    
                    # Fetch full details including markdown content
                    with st.spinner("Loading document details..."):
                        selected_row = get_document_details(selected_file_name)
                    
                    if selected_row is not None:
                        # Display key details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**File Information:**")
                            for col in ['file_name', 'volume_path', 'directory', 'file_extension']:
                                if col in selected_row and pd.notna(selected_row[col]):
                                    st.write(f"- **{col.replace('_', ' ').title()}:** {selected_row[col]}")
                        
                        with col2:
                            st.write("**Processing Information:**")
                            for col in ['extraction_success', 'page_count', 'character_count', 'processing_time_seconds']:
                                if col in selected_row and pd.notna(selected_row[col]):
                                    st.write(f"- **{col.replace('_', ' ').title()}:** {selected_row[col]}")
                        
                        # Show error message if extraction failed
                        if 'error_message' in selected_row and pd.notna(selected_row['error_message']) and selected_row['error_message']:
                            st.write("**Error Message:**")
                            st.error(selected_row['error_message'])
                        
                        # Show markdown content and PDF side by side if available
                        if 'markdown_content' in selected_row and pd.notna(selected_row['markdown_content']) and selected_row['markdown_content']:
                            st.write("**Document Comparison - Markdown vs PDF**")
                            
                            # Create two columns for side-by-side display
                            pdf_col, markdown_col = st.columns(2)
                            
                            with pdf_col:
                                st.subheader("Original PDF")
                                # Get the PDF file path from volume_path
                                pdf_path = selected_row.get('volume_path', '')
                                if pdf_path:
                                    with st.spinner("Loading PDF..."):
                                        local_path = download_pdf_file(pdf_path)
                                    if local_path:
                                        pdf_viewer(local_path, width=450, height=600)
                                        # Clean up temp file after displaying
                                        try:
                                            os.unlink(local_path)
                                        except:
                                            pass
                                    else:
                                        st.error("Error loading PDF from volume.")
                                else:
                                    st.warning("PDF path not available.")
                            
                            with markdown_col:
                                st.subheader("Extracted Markdown")
                                # Display markdown in a scrollable container
                                st.markdown(
                                    f"""
                                    <div style="height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                                    {selected_row['markdown_content']}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.error("Could not load details for the selected document.")
        else:
            st.warning("No data columns found to display.")
            
    else:
        st.warning("No data found in the document_markdown table. Please check your SQL warehouse connection and table permissions.")
        st.info("Make sure the DATABRICKS_SQL_WAREHOUSE environment variable is set correctly.")

# Page 3: View VLM Extraction
elif page == "View VLM Extraction":
    st.title("VLM Extraction")
    st.markdown("**Extract and view OCR content from document pages**")
    st.markdown(f"**SQL Warehouse:** `{SQL_WAREHOUSE}` (set via $DATABRICKS_SQL_WAREHOUSE env var)")
    
    # Add refresh button for the table data
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh VLM Data") or 'vlm_data' not in st.session_state:
            logger.info("Refreshing VLM document data from SQL warehouse")
            st.session_state['vlm_data'] = get_vlm_document_list()
    
    # Get the data
    vlm_df = st.session_state.get('vlm_data', pd.DataFrame())
    
    if not vlm_df.empty:
        st.subheader("Document Processing Results")
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(vlm_df))
        with col2:
            total_pages = vlm_df['total_pages'].sum() if 'total_pages' in vlm_df.columns else 0
            st.metric("Total Pages", int(total_pages) if pd.notna(total_pages) else 0)
        with col3:
            total_size_mb = (vlm_df['file_size_bytes'].sum() / (1024*1024)) if 'file_size_bytes' in vlm_df.columns else 0
            st.metric("Total Size", f"{total_size_mb:.1f} MB" if pd.notna(total_size_mb) else "N/A")
        
        # Document selection
        st.subheader("Select Document")
        
        # Display documents table
        display_vlm_df = vlm_df[['source_filename', 'total_pages']].copy()
        st.dataframe(
            display_vlm_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_filename": "Document Name",
                "total_pages": "Total Pages"
            }
        )
        
        # Document selection dropdown
        selected_document = st.selectbox(
            "Choose a document to view:",
            options=vlm_df['source_filename'].tolist(),
            format_func=lambda x: x if x else "Select a document..."
        )
        
        if selected_document:
            # Load all pages for the selected document
            with st.spinner("Loading document pages..."):
                pages_df = get_vlm_document_pages(selected_document)
            
            if not pages_df.empty:
                # Page navigation
                st.subheader(f"Document: {selected_document}")
                
                max_pages = len(pages_df)
                if max_pages > 0:
                    # Page selection
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        current_page = st.selectbox(
                            f"Select Page (1 to {max_pages}):",
                            options=range(1, max_pages + 1),
                            index=0,
                            format_func=lambda x: f"Page {x}"
                        )
                    
                    # Get the current page data
                    page_data = pages_df[pages_df['page_number'] == current_page]
                    if not page_data.empty:
                        page_row = page_data.iloc[0]
                        
                        # Display page information
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Page", f"{current_page}/{max_pages}")
                        with col2:
                            st.metric("OCR Model", page_row.get('ocr_model', 'N/A'))
                        with col3:
                            if 'ocr_timestamp' in page_row and pd.notna(page_row['ocr_timestamp']):
                                st.metric("OCR Date", str(page_row['ocr_timestamp'])[:10])
                        with col4:
                            char_count = len(page_row.get('ocr_text', '')) if pd.notna(page_row.get('ocr_text')) else 0
                            st.metric("Text Length", f"{char_count} chars")
                        
                        # Side-by-side display
                        st.markdown("---")
                        image_col, text_col = st.columns(2)
                        
                        with image_col:
                            st.subheader("Page Image")
                            if 'page_image_png' in page_row and pd.notna(page_row['page_image_png']):
                                try:
                                    # Convert binary PNG data to displayable image
                                    import base64
                                    from io import BytesIO
                                    
                                    # Get the binary data
                                    image_binary = page_row['page_image_png']
                                    
                                    # Display the image
                                    st.image(
                                        image_binary,
                                        caption=f"Page {current_page}",
                                        use_column_width=True
                                    )
                                except Exception as e:
                                    st.error(f"Error displaying image: {e}")
                                    logger.error(f"Error displaying page image: {e}")
                            else:
                                st.warning("No image data available for this page.")
                        
                        with text_col:
                            st.subheader("OCR Text")
                            if 'ocr_text' in page_row and pd.notna(page_row['ocr_text']) and page_row['ocr_text']:
                                # Display OCR text in a scrollable text area
                                ocr_text = str(page_row['ocr_text'])
                                
                                # Show character count for debugging
                                st.caption(f"Text length: {len(ocr_text)} characters")
                                
                                # Use st.text_area for better text display
                                st.text_area(
                                    "OCR Content:",
                                    value=ocr_text,
                                    height=600,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                            else:
                                st.warning("No OCR text available for this page.")
                        
                        # Additional page navigation buttons
                        st.markdown("---")
                        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
                        
                        with nav_col1:
                            if st.button("⏮️ First", disabled=(current_page == 1)):
                                st.rerun()
                        with nav_col2:
                            if st.button("⬅️ Previous", disabled=(current_page == 1)):
                                st.rerun()
                        with nav_col3:
                            st.write(f"Page {current_page}")
                        with nav_col4:
                            if st.button("➡️ Next", disabled=(current_page == max_pages)):
                                st.rerun()
                        with nav_col5:
                            if st.button("⏭️ Last", disabled=(current_page == max_pages)):
                                st.rerun()
                    else:
                        st.error(f"No data found for page {current_page}")
                else:
                    st.warning("No pages found for the selected document.")
            else:
                st.warning("No page data found for the selected document.")
    else:
        st.warning("No data found in the document_store_ocr table. Please check your SQL warehouse connection and table permissions.")
        st.info("Make sure the DATABRICKS_SQL_WAREHOUSE environment variable is set correctly.")