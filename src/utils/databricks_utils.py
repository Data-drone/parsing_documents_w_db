from pyspark.dbutils import DBUtils

def get_username_from_email(dbutils: DBUtils):
    """
    Extract username from email address in Databricks notebook context.
    """
    try:
        user_email = (dbutils.notebook.entry_point
                                .getDbutils()
                                .notebook()
                                .getContext()
                                .tags().apply("user")
            )
        
        if not user_email or '@' not in user_email:
            raise ValueError("Invalid email address")
            
        username = user_email.split('@')[0].replace('.', '_')
        
        return username
        
    except Exception as e:
        raise ValueError(f"Failed to get username: {str(e)}")