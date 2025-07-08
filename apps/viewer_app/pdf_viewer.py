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
import json
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
VOLUME_PATH = os.environ.get("DATABRICKS_VOLUME", "")
SQL_WAREHOUSE = os.environ.get("DATABRICKS_SQL_WAREHOUSE", "")
DATABRICKS_APP_PRINCIPAL = os.environ.get("DATABRICKS_CLIENT_ID", "")
DATABRICKS_APP_PRINCIPAL_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET", "")
HOST_PATH = os.environ.get("DATABRICKS_HOST", "")

### need to fix how to get this properly
URL_BASEPATH = "https://adb-984752964297111.11.azuredatabricks.net" # f"{urlparse(HOST_PATH).scheme}://{urlparse(HOST_PATH).netloc}"

# Databricks configuration
cfg = Config()

st.set_page_config(page_title="Databricks Volume PDF Viewer", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Select Page",
        ["Home", "View Volumes", "View Markdown Extraction", "View VLM Extraction", "Vector Search"],
        index=0
    )
    st.divider()

# List files in the given volume path
def list_files_in_volume():
    # Use configured volume path if available, otherwise fall back to environment variable
    volume_path = st.session_state.get('constructed_volume_path', VOLUME_PATH)
    
    w = WorkspaceClient()
    try:
        files = [f.path for f in w.files.list_directory_contents(volume_path)]
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing files in volume {volume_path}: {e}")
        return []

# Unity Catalog browsing functions
def get_catalogs():
    """Get list of available UC catalogs"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            cursor.execute("SHOW CATALOGS")
            result = cursor.fetchall_arrow().to_pandas()
            return result['catalog'].tolist() if 'catalog' in result.columns else []
    except Exception as e:
        logger.error(f"Error listing catalogs: {e}")
        return []

def get_schemas(catalog):
    """Get list of schemas in a catalog"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW SCHEMAS IN {catalog}")
            result = cursor.fetchall_arrow().to_pandas()
            return result['databaseName'].tolist() if 'databaseName' in result.columns else []
    except Exception as e:
        logger.error(f"Error listing schemas in {catalog}: {e}")
        return []

def get_tables(catalog, schema):
    """Get list of tables in a catalog.schema"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
            result = cursor.fetchall_arrow().to_pandas()
            return result['tableName'].tolist() if 'tableName' in result.columns else []
    except Exception as e:
        logger.error(f"Error listing tables in {catalog}.{schema}: {e}")
        return []

def get_volumes(catalog, schema):
    """Get list of volumes in a catalog.schema"""
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW VOLUMES IN {catalog}.{schema}")
            result = cursor.fetchall_arrow().to_pandas()
            return result['volume_name'].tolist() if 'volume_name' in result.columns else []
    except Exception as e:
        logger.error(f"Error listing volumes in {catalog}.{schema}: {e}")
        return []

def get_vector_search_endpoints():
    """Get list of vector search endpoints"""
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient(disable_notice=True,
                                 workspace_url=URL_BASEPATH,
                                 service_principal_client_id=DATABRICKS_APP_PRINCIPAL,
                                 service_principal_client_secret=DATABRICKS_APP_PRINCIPAL_SECRET)
        endpoints = vsc.list_endpoints()
        return [endpoint['name'] for endpoint in endpoints.get('endpoints', [])] if endpoints else []
    except ImportError:
        logger.error("databricks-vectorsearch package not installed. Run: pip install databricks-vectorsearch")
        return []
    except Exception as e:
        logger.error(f"Error listing vector search endpoints: {e}")
        return []

def get_vector_search_indexes(endpoint_name):
    """Get list of vector search indexes for an endpoint"""
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient(disable_notice=True,
                                 workspace_url=URL_BASEPATH,
                                 service_principal_client_id=DATABRICKS_APP_PRINCIPAL,
                                 service_principal_client_secret=DATABRICKS_APP_PRINCIPAL_SECRET)
        
        indexes = vsc.list_indexes(name=endpoint_name)
        return [index['name'] for index in indexes.get('vector_indexes', [])] if indexes else []
    except ImportError:
        logger.error("databricks-vectorsearch package not installed. Run: pip install databricks-vectorsearch")
        return []
    except Exception as e:
        logger.error(f"Error listing vector search indexes for endpoint {endpoint_name}: {e}")
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
    if not st.session_state.get('config_saved'):
        return pd.DataFrame()
    
    # Get table name from session state
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('markdown_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = f"""
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
            FROM {full_table_name}
            ORDER BY processing_timestamp DESC
            """
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying document_markdown table {full_table_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Query full details for a specific document (only what we need for display)
def get_document_details(file_name):
    """Get full details for a specific document including markdown content"""
    if not st.session_state.get('config_saved'):
        return None
    
    # Get table name from session state
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('markdown_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = f"""
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
            FROM {full_table_name}
            WHERE file_name = ?
            """
            cursor.execute(query, (file_name,))
            result = cursor.fetchall_arrow().to_pandas()
            return result.iloc[0] if len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error querying document details from {full_table_name}: {e}")
        return None

# VLM Extraction Functions
def get_vlm_document_list():
    """Get distinct source filenames from the document_store_ocr table"""
    if not st.session_state.get('config_saved'):
        return pd.DataFrame()
    
    # Get table name from session state
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('vlm_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = f"""
            SELECT DISTINCT 
                source_filename,
                MAX(total_pages) as total_pages,
                MAX(file_size_bytes) as file_size_bytes,
                MAX(processing_timestamp) as latest_processing
            FROM {full_table_name}
            WHERE source_filename IS NOT NULL
            GROUP BY source_filename
            ORDER BY latest_processing DESC
            """
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying VLM document list from {full_table_name}: {e}")
        return pd.DataFrame()

def get_vlm_document_pages(source_filename):
    """Get all pages for a specific document from the document_store_ocr table"""
    if not st.session_state.get('config_saved'):
        return pd.DataFrame()
    
    # Get table name from session state
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('vlm_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    
    try:
        conn = get_connection(SQL_WAREHOUSE)
        with conn.cursor() as cursor:
            query = f"""
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
            FROM {full_table_name}
            WHERE source_filename = ?
            ORDER BY page_number ASC
            """
            cursor.execute(query, (source_filename,))
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        logger.error(f"Error querying VLM document pages from {full_table_name}: {e}")
        return pd.DataFrame()

# Vector Search Functions
def query_vector_search(query_text, num_results=5):
    """Query the vector search index"""
    if not st.session_state.get('config_saved'):
        return []
    
    endpoint_name = st.session_state.get('selected_vs_endpoint', '')
    index_name = st.session_state.get('selected_vs_index', '')
    
    if not endpoint_name or not index_name:
        return []
    
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient(disable_notice=True,
                                 workspace_url=URL_BASEPATH,
                                 service_principal_client_id=DATABRICKS_APP_PRINCIPAL,
                                 service_principal_client_secret=DATABRICKS_APP_PRINCIPAL_SECRET)
        
        # Get the index object
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        
        index_columns = [x['key'] for x in index.scan(num_results=1)['data'][0]['fields']]
        
        # Query the vector search index
        results = index.similarity_search(
            query_text=query_text,
            columns=index_columns,  # Get all columns
            num_results=num_results
        )
        
        # Convert results to a more manageable format
        ### this bit needs fixing
        
        formatted_results = []
        if results and 'result' in results and 'data_array' in results['result']:
            for item in results['result']['data_array']:
                list_of_dicts = [dict(zip(index_columns, row)) for row in results['result']['data_array']]
    
                for item in list_of_dicts:
                    formatted_results.append(item)
        
        return formatted_results
    except ImportError:
        logger.error("databricks-vectorsearch package not installed. Run: pip install databricks-vectorsearch")
        return []
    except Exception as e:
        logger.error(f"Error querying vector search index {index_name}: {e}")
        return []

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

# Page 0: Home - Configuration
if page == "Home":
    st.title("Databricks Document Processing Hub")
    st.markdown("**Configure your Unity Catalog connection and data processing pipeline**")
    st.markdown(f"**Default Volume Path:** `{VOLUME_PATH}` | **SQL Warehouse:** `{SQL_WAREHOUSE}`")
    
    # Data Flow Visualization
    st.subheader("📊 Data Processing Pipeline")
    
    # Responsive pipeline visualization
    col1, arrow1, col2, arrow2, col3 = st.columns([3, 1, 3, 1, 3])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; 
                    color: white; min-height: 120px; display: flex; 
                    flex-direction: column; justify-content: center;">
            <h3 style="color: white; margin: 0;">📁 Unity Catalog Volume</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">PDF Files Storage</p>
            <code style="background: rgba(255,255,255,0.2); padding: 5px; 
                        border-radius: 5px; color: white;">/Volumes/cat/sch/vol</code>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow1:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 120px;">
            <div style="font-size: 24px;">→</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; 
                    color: white; min-height: 120px; display: flex; 
                    flex-direction: column; justify-content: center;">
            <h3 style="color: white; margin: 0;">📄 Markdown Table</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Document Text + Metadata</p>
            <small style="opacity: 0.8;">Full text extraction & analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow2:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 120px;">
            <div style="font-size: 24px;">→</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; 
                    color: white; min-height: 120px; display: flex; 
                    flex-direction: column; justify-content: center;">
            <h3 style="color: white; margin: 0;">🖼️ VLM/OCR Table</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Page Images + OCR Text</p>
            <small style="opacity: 0.8;">Visual content extraction</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed descriptions below - aligned with pipeline boxes
    st.markdown("<br>", unsafe_allow_html=True)
    desc_col1, desc_spacer1, desc_col2, desc_spacer2, desc_col3 = st.columns([3, 1, 3, 1, 3])
    
    with desc_col1:
        st.info("""
        **📁 Source Volume**
        
        Raw PDF files stored in Unity Catalog volumes. This is your document repository where all source files are maintained with proper governance and access controls.
        
        • File storage & organization  
        • Access control & permissions  
        • Version management  
        """)
    
    with desc_col2:
        st.info("""
        **📄 Text Processing**
        
        Extracted markdown content with comprehensive document metadata. This stage converts PDFs into structured text data ready for analysis and search.
        
        • Full text extraction  
        • Document metadata  
        • Processing statistics  
        """)
    
    with desc_col3:
        st.info("""
        **🖼️ Visual Processing**
        
        Page-by-page image extraction with OCR text content. This enables visual document analysis and provides backup text extraction methods.
        
        • Page image capture  
        • OCR text extraction  
        • Visual content analysis  
        """)
    
    st.markdown("---")
    
    # Configuration section
    st.subheader("🔧 Configuration")
    
    # Test connection first
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔗 Test Connection"):
            try:
                conn = get_connection(SQL_WAREHOUSE)
                st.success("✅ SQL Warehouse connection successful!")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.stop()
    
    # Catalog selection
    st.subheader("📁 Unity Catalog Selection")
    
    # Load catalogs
    with st.spinner("Loading catalogs..."):
        catalogs = get_catalogs()
    
    if catalogs:
        selected_catalog = st.selectbox(
            "Select Catalog:",
            options=catalogs,
            index=catalogs.index(st.session_state.get('selected_catalog', catalogs[0])) if st.session_state.get('selected_catalog') in catalogs else 0
        )
        st.session_state['selected_catalog'] = selected_catalog
        
        # Load schemas for selected catalog
        with st.spinner(f"Loading schemas in {selected_catalog}..."):
            schemas = get_schemas(selected_catalog)
        
        if schemas:
            selected_schema = st.selectbox(
                "Select Schema:",
                options=schemas,
                index=schemas.index(st.session_state.get('selected_schema', schemas[0])) if st.session_state.get('selected_schema') in schemas else 0
            )
            st.session_state['selected_schema'] = selected_schema
            
            # Volume Configuration Section
            st.subheader("📁 Volume Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Volume Schema Selection:**")
                volume_schema = st.selectbox(
                    "Schema for Volume:",
                    options=schemas,
                    index=schemas.index(st.session_state.get('volume_schema', selected_schema)) if st.session_state.get('volume_schema') in schemas else schemas.index(selected_schema),
                    key="volume_schema_select"
                )
                st.session_state['volume_schema'] = volume_schema
            
            with col2:
                st.markdown("**Volume Selection:**")
                # Load volumes for selected catalog.schema
                with st.spinner(f"Loading volumes in {selected_catalog}.{volume_schema}..."):
                    volumes = get_volumes(selected_catalog, volume_schema)
                
                if volumes:
                    selected_volume = st.selectbox(
                        "Select Volume:",
                        options=volumes,
                        index=volumes.index(st.session_state.get('selected_volume', volumes[0])) if st.session_state.get('selected_volume') in volumes else 0
                    )
                    st.session_state['selected_volume'] = selected_volume
                    
                    # Construct full volume path
                    volume_path = f"/Volumes/{selected_catalog}/{volume_schema}/{selected_volume}"
                    st.session_state['constructed_volume_path'] = volume_path
                    st.code(f"Volume Path: {volume_path}")
                else:
                    st.warning(f"No volumes found in {selected_catalog}.{volume_schema}")
                    # Fallback to environment variable
                    st.session_state['constructed_volume_path'] = VOLUME_PATH
                    st.info(f"Using environment volume path: {VOLUME_PATH}")
            
            # Load tables for selected catalog.schema
            with st.spinner(f"Loading tables in {selected_catalog}.{selected_schema}..."):
                tables = get_tables(selected_catalog, selected_schema)
            
            if tables:
                st.subheader("📊 Table Selection")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Markdown Extraction Table:**")
                    markdown_table = st.selectbox(
                        "Table for markdown extraction:",
                        options=tables,
                        index=tables.index(st.session_state.get('markdown_table', tables[0])) if st.session_state.get('markdown_table') in tables else 0,
                        key="markdown_table_select"
                    )
                    st.session_state['markdown_table'] = markdown_table
                    
                    # Show selected table name
                    markdown_full_name = f"{selected_catalog}.{selected_schema}.{markdown_table}"
                    st.code(markdown_full_name)
                
                with col2:
                    st.markdown("**VLM/OCR Extraction Table:**")
                    vlm_table = st.selectbox(
                        "Table for VLM/OCR extraction:",
                        options=tables,
                        index=tables.index(st.session_state.get('vlm_table', tables[0])) if st.session_state.get('vlm_table') in tables else 0,
                        key="vlm_table_select"
                    )
                    st.session_state['vlm_table'] = vlm_table
                    
                    # Show selected table name
                    vlm_full_name = f"{selected_catalog}.{selected_schema}.{vlm_table}"
                    st.code(vlm_full_name)
                
                # Vector Search Configuration Section
                st.subheader("🔍 Vector Search Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Vector Search Endpoint:**")
                    # Load vector search endpoints
                    with st.spinner("Loading vector search endpoints..."):
                        vs_endpoints = get_vector_search_endpoints()
                    
                    if vs_endpoints:
                        selected_vs_endpoint = st.selectbox(
                            "Select Vector Search Endpoint:",
                            options=vs_endpoints,
                            index=vs_endpoints.index(st.session_state.get('selected_vs_endpoint', vs_endpoints[0])) if st.session_state.get('selected_vs_endpoint') in vs_endpoints else 0,
                            key="vs_endpoint_select"
                        )
                        st.session_state['selected_vs_endpoint'] = selected_vs_endpoint
                        st.code(f"Endpoint: {selected_vs_endpoint}")
                    else:
                        st.warning("No vector search endpoints found. Make sure the `databricks-vectorsearch` package is installed.")
                        st.code("pip install databricks-vectorsearch")
                        selected_vs_endpoint = None
                
                with col2:
                    st.markdown("**Vector Search Index:**")
                    if selected_vs_endpoint:
                        # Load vector search indexes for selected endpoint
                        with st.spinner(f"Loading indexes for {selected_vs_endpoint}..."):
                            vs_indexes = get_vector_search_indexes(selected_vs_endpoint)
                        
                        if vs_indexes:
                            selected_vs_index = st.selectbox(
                                "Select Vector Search Index:",
                                options=vs_indexes,
                                index=vs_indexes.index(st.session_state.get('selected_vs_index', vs_indexes[0])) if st.session_state.get('selected_vs_index') in vs_indexes else 0,
                                key="vs_index_select"
                            )
                            st.session_state['selected_vs_index'] = selected_vs_index
                            st.code(f"Index: {selected_vs_index}")
                        else:
                            st.warning(f"No indexes found for endpoint {selected_vs_endpoint}")
                    else:
                        st.warning("Please select a vector search endpoint first.")
                
                # Save configuration
                st.markdown("---")
                if st.button("💾 Save Configuration"):
                    st.session_state['config_saved'] = True
                    st.success("✅ Configuration saved! You can now use the other tabs.")
                
                # Show current configuration
                if st.session_state.get('config_saved'):
                    st.success("✅ Configuration is saved and ready to use!")
                    
                    with st.expander("📋 Current Configuration", expanded=False):
                        st.write(f"**Catalog:** {selected_catalog}")
                        st.write(f"**Schema:** {selected_schema}")
                        st.write(f"**Volume Schema:** {volume_schema}")
                        st.write(f"**Volume Path:** {st.session_state.get('constructed_volume_path', VOLUME_PATH)}")
                        st.write(f"**Markdown Table:** {markdown_full_name}")
                        st.write(f"**VLM Table:** {vlm_full_name}")
                        st.write(f"**Vector Search Endpoint:** {st.session_state.get('selected_vs_endpoint', 'Not configured')}")
                        st.write(f"**Vector Search Index:** {st.session_state.get('selected_vs_index', 'Not configured')}")
                        st.write(f"**SQL Warehouse:** {SQL_WAREHOUSE}")
            else:
                st.warning(f"No tables found in {selected_catalog}.{selected_schema}")
        else:
            st.warning(f"No schemas found in catalog '{selected_catalog}'")
    else:
        st.error("No catalogs found. Please check your SQL warehouse connection and permissions.")
    
    # Navigation hint
    if not st.session_state.get('config_saved'):
        st.info("👆 Please configure your catalog, volume, and tables above, then save the configuration to use the other tabs.")

# Page 1: View Volumes
elif page == "View Volumes":
    st.title("Databricks Volume PDF Viewer (SDK)")
    
    # Show configured volume path
    volume_path = st.session_state.get('constructed_volume_path', VOLUME_PATH)
    st.markdown(f"**Current Volume Path:** `{volume_path}`")
    
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
    # Check if configuration is saved
    if not st.session_state.get('config_saved'):
        st.warning("⚠️ Please configure your catalog and tables on the Home page first.")
        st.stop()
    
    st.title("Markdown Extraction")
    st.markdown("**Extract and view markdown content from PDF files**")
    
    # Show current table configuration
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('markdown_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    st.markdown(f"**Table:** `{full_table_name}` | **SQL Warehouse:** `{SQL_WAREHOUSE}`")
    
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
    # Check if configuration is saved
    if not st.session_state.get('config_saved'):
        st.warning("⚠️ Please configure your catalog and tables on the Home page first.")
        st.stop()
    
    st.title("VLM Extraction")
    st.markdown("**Extract and view OCR content from document pages**")
    
    # Show current table configuration
    catalog = st.session_state.get('selected_catalog', '')
    schema = st.session_state.get('selected_schema', '')
    table = st.session_state.get('vlm_table', '')
    full_table_name = f"{catalog}.{schema}.{table}"
    st.markdown(f"**Table:** `{full_table_name}` | **SQL Warehouse:** `{SQL_WAREHOUSE}`")
    
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

# Page 4: Vector Search
elif page == "Vector Search":
    # Check if configuration is saved
    if not st.session_state.get('config_saved'):
        st.warning("⚠️ Please configure your catalog and tables on the Home page first.")
        st.stop()
    
    st.title("Vector Search")
    st.markdown("**Search through document content using semantic similarity**")
    
    # Show current vector search configuration
    endpoint_name = st.session_state.get('selected_vs_endpoint', '')
    index_name = st.session_state.get('selected_vs_index', '')
    
    if not endpoint_name or not index_name:
        st.warning("⚠️ Please configure your Vector Search endpoint and index on the Home page first.")
        st.stop()
    
    st.markdown(f"**Endpoint:** `{endpoint_name}` | **Index:** `{index_name}`")
    
    # Search interface
    st.subheader("🔍 Semantic Search")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query_text = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'financial performance metrics', 'risk management strategies', 'customer satisfaction data'",
            help="Enter natural language queries to find semantically similar content"
        )
    
    with col2:
        num_results = st.selectbox(
            "Results:",
            options=[5, 10, 15, 20],
            index=0
        )
    
    # Search button and results
    if st.button("🔍 Search", type="primary") or query_text:
        if query_text.strip():
            with st.spinner("Searching..."):
                results = query_vector_search(query_text, num_results)
            
            if results:
                st.subheader(f"📋 Search Results ({len(results)} found)")
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}", expanded=(i <= 3)):
                        # Display all fields in the result
                        for key, value in result.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            
                            # Handle different value types
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            elif isinstance(value, str) and len(value) > 500:
                                # For long text, show in a text area
                                st.text_area(f"{key}", value=value, height=200, disabled=True, label_visibility="collapsed")
                            else:
                                st.write(value)
                            
                            st.markdown("---")
                
                # Simple statistics
                st.markdown("---")
                st.metric("Total Results", len(results))
                
            else:
                st.warning("No results found for your query. Try different keywords or check your vector search configuration.")
                # Check if it's a package issue
                try:
                    from databricks.vector_search.client import VectorSearchClient
                except ImportError:
                    st.error("❌ `databricks-vectorsearch` package not installed. Install it with: `pip install databricks-vectorsearch`")
        else:
            st.info("Please enter a search query to get started.")
    
    # Help section
    with st.expander("💡 Search Tips", expanded=False):
        st.markdown("""
        **How to write effective search queries:**
        
        • **Use natural language**: Write queries as you would ask a question
        • **Be specific**: Include key terms and concepts you're looking for
        • **Use context**: Add context words to narrow down results
        • **Try synonyms**: If no results, try alternative phrasings
        
        **Example queries:**
        - "What are the main financial risks mentioned?"
        - "Customer satisfaction scores and feedback"
        - "Product performance metrics Q4"
        - "Regulatory compliance requirements"
        """)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        st.markdown("**Vector Search Configuration:**")
        st.code(f"Endpoint: {endpoint_name}")
        st.code(f"Index: {index_name}")
        
        st.markdown("**Requirements:**")
        st.info("Make sure `databricks-vectorsearch` package is installed: `pip install databricks-vectorsearch`")
        
        if st.button("🔄 Test Connection"):
            try:
                from databricks.vector_search.client import VectorSearchClient
                vsc = VectorSearchClient(disable_notice=True,
                                         workspace_url=URL_BASEPATH,
                                         service_principal_client_id=DATABRICKS_APP_PRINCIPAL,
                                         service_principal_client_secret=DATABRICKS_APP_PRINCIPAL_SECRET)
                # Try to get index info
                index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
                index_info = index.describe()
                st.success("✅ Vector search connection successful!")
                st.json({
                    "index_type": index_info.get("index_type", "N/A"),
                    "status": index_info.get("status", {}).get("detailed_state", "N/A"),
                    "primary_key": index_info.get("primary_key", "N/A")
                })
            except ImportError:
                st.error("❌ `databricks-vectorsearch` package not installed. Run: `pip install databricks-vectorsearch`")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")