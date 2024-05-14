# in myfunc.prompts.py
import json
import mysql.connector
import os
import tiktoken

from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai.chat_models import ChatOpenAI

from mysql.connector import Error

from myfunc.varvars_dicts import work_vars


# in myfunc.prompts.py
class PromptDatabase:
    """
    A class to interact with a MySQL database for storing and retrieving prompt templates.
    """
    def __init__(self, host=None, user=None, password=None, database=None):
        """
        Initializes the connection details for the database, with the option to use environment variables as defaults.
        """
        self.host = host if host is not None else os.getenv('DB_HOST')
        self.user = user if user is not None else os.getenv('DB_USER')
        self.password = password if password is not None else os.getenv('DB_PASSWORD')
        self.database = database if database is not None else os.getenv('DB_NAME')
        self.conn = None
        self.cursor = None
        
    def __enter__(self):
        """
        Establishes the database connection and returns the instance itself when entering the context.
        """
        self.conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the database connection and cursor when exiting the context.
        Handles any exceptions that occurred within the context.
        """
        if self.cursor is not None:
            self.cursor.close()
        if self.conn is not None:
            self.conn.close()
        # Handle exception if needed, can log or re-raise exceptions based on requirements
        if exc_type or exc_val or exc_tb:
            # Optionally log or handle exception
            pass

    # !!!! poziva ga osnovni metod za pretragu u ostalim .py - get_prompts_by_names 
    def query_sql_prompt_strings(self, prompt_names):
        """
        Fetches the existing prompt strings for a given list of prompt names, maintaining the order of prompt_names.
        """
        order_clause = "ORDER BY CASE PromptName "
        for idx, name in enumerate(prompt_names):
            order_clause += f"WHEN %s THEN {idx} "
        order_clause += "END"

        query = f"""
        SELECT PromptString FROM PromptStrings
        WHERE PromptName IN ({','.join(['%s'] * len(prompt_names))})
        """ + order_clause

        params = tuple(prompt_names) + tuple(prompt_names)  # prompt_names repeated for both IN and ORDER BY
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        return [result[0] for result in results] if results else []


    # za odabir za selectbox i za funkcije unosa i editovanja - osnovna funkcija
    def get_records(self, query, params=None):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.__enter__()
            self.cursor.execute(query, params)
            records = self.cursor.fetchall()
            return records
        except Error as e:
            
            return []
        
    # opsta funkcija za prikaz polja za selectbox - koristi get_recoirds za pripremu
    def get_records_from_column(self, table, column):
        """
        Fetch records from a specified column in a specified table.
        """
        query = f"SELECT DISTINCT {column} FROM {table}"
        records = self.get_records(query)
        return [record[0] for record in records] if records else []
    
    # !!!! osnovni metod za pretragu u ostalim .py    
    def get_prompts_by_names(self, variable_names, prompt_names):
        prompt_strings = self.query_sql_prompt_strings(prompt_names)
        prompt_variables = dict(zip(variable_names, prompt_strings))
        return prompt_variables

    # za prikaz cele tabele kao info prilikom unosa i editovanja koristi kasnije df
    def get_all_records_from_table(self, table_name):
        """
        Fetch all records and all columns for a given table.
        :param table_name: The name of the table from which to fetch records.
        :return: A pandas DataFrame with all records and columns from the specified table.
        """
        query = f"SELECT * FROM {table_name}"
        try:
            self.cursor.execute(query)
            records = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return records, columns
        except Exception as e:
            print(f"Failed to fetch records: {e}")
            return [],[]  # Return an empty DataFrame in case of an error
    
    # prikazuje tabelu promptstrings za dati username
    def get_prompts_for_username(self, username):
        """
        Fetch all prompt texts and matching variable names for a given username.
        :param username: The username (or partial username) for which to fetch prompt texts and variable names.
        :return: A pandas DataFrame with the prompt texts and matching variable names.
        """
        query = """
        SELECT ps.PromptName, ps.PromptString, pv.VariableName, pf.Filename, pf.FilePath,u.Username 
        FROM PromptStrings ps
        JOIN PromptVariables pv ON ps.VariableID = pv.VariableID
        JOIN PythonFiles pf ON ps.VariableFileID = pf.FileID
        JOIN Users u ON ps.UserID = u.UserID
        WHERE u.Username LIKE %s
        """
        params = (f"%{username}%",)
        records = self.get_records(query, params)
        return records
        
    # za unos u pomocne tabele Users, PythonFiles, PromptVariables
    def add_record(self, table, **fields):
        columns = ', '.join(fields.keys())
        placeholders = ', '.join(['%s'] * len(fields))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        try:
            self.cursor.execute(query, tuple(fields.values()))
            self.conn.commit()
            return self.cursor.lastrowid
        except Error as e:
            self.conn.rollback()
            print(f"Error in add_record: {e}")
            return None
            
    # za unos u tabelu promptstrings
    def add_new_record(self, username, filename, variablename, promptstring, promptname, comment):
        """
        Adds a new record to the database, handling the relationships between users, files, variables, and prompts.
        """
        try:
            # Fetch UserID based on username
            self.cursor.execute("SELECT UserID FROM Users WHERE Username = %s", (username,))
            user_result = self.cursor.fetchone()
            user_id = user_result[0] if user_result else None

            # Fetch VariableID based on variablename
            self.cursor.execute("SELECT VariableID FROM PromptVariables WHERE VariableName = %s", (variablename,))
            variable_result = self.cursor.fetchone()
            variable_id = variable_result[0] if variable_result else None

            # Fetch FileID based on filename
            self.cursor.execute("SELECT FileID FROM PythonFiles WHERE Filename = %s", (filename,))
            file_result = self.cursor.fetchone()
            file_id = file_result[0] if file_result else None

            # Ensure all IDs are found
            if not all([user_id, variable_id, file_id]):
                return "Error: Missing UserID, VariableID, or VariableFileID."

            # Correctly include FileID in the insertion command
            self.cursor.execute(
                "INSERT INTO PromptStrings (PromptString, PromptName, Comment, UserID, VariableID, VariableFileID) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE Comment=VALUES(Comment), UserID=VALUES(UserID), VariableID=VALUES(VariableID), VariableFileID=VALUES(VariableFileID);",
                (promptstring, promptname, comment, user_id, variable_id, file_id)
            )

            self.conn.commit()
            return "Record added successfully."
        except Exception as e:
            self.conn.rollback()
            return f"Failed to add the record: {e}"

    def update_record(self, table, fields, condition):
        """
        Updates records in the specified table based on a condition.
    
        :param table: The name of the table to update.
        :param fields: A dictionary of column names and their new values.
        :param condition: A tuple containing the condition string and its values (e.g., ("UserID = %s", [user_id])).
        """
        set_clause = ', '.join([f"{key} = %s" for key in fields.keys()])
        values = list(fields.values()) + condition[1]
    
        query = f"UPDATE {table} SET {set_clause} WHERE {condition[0]}"
    
        try:
            self.cursor.execute(query, values)
            self.conn.commit()
            return "Uspesno dodat slog"  # Returns the number of rows affected
        except Exception as e:
            self.conn.rollback()
            return f"Error in update_record: {e}"
        
    # za brisanje recorda u promptstrings   
    def delete_prompt_by_name(self, promptname):
        """
        Delete a prompt record from PromptStrings by promptname.
        This method also handles deletions or updates in related tables if necessary.
        """
        # Optional: Handle related data in other tables before deleting the prompt.
        # This could involve setting foreign keys to NULL or cascading deletions, depending on your schema.
        
        # Delete the prompt from PromptStrings
        delete_query = "DELETE FROM PromptStrings WHERE PromptName = %s;"
        try:
            if self.conn is None or not self.conn.is_connected():
                self.__enter__()
            self.cursor.execute(delete_query, (promptname))
            self.conn.commit()  # Commit the transaction to finalize the deletion
            return f"Prompt '{promptname}' deleted successfully."
        except Error as e:
            self.conn.rollback()  # Rollback in case of error
            return f"Error deleting prompt '{promptname}': {e}"
    
    # za update u promptstrings
    def update_prompt_record(self, promptname, new_promptstring, new_comment):
        """
        Updates the PromptString and Comment fields of an existing prompt record identified by PromptName.
    
        :param promptname: The name of the prompt to update.
        :param new_promptstring: The new value for the PromptString field.
        :param new_comment: The new value for the Comment field.
        """
        try:
            if self.conn is None or not self.conn.is_connected():
                self.__enter__()
        
            # Prepare the SQL update statement
            sql_update_query = """
            UPDATE PromptStrings 
            SET PromptString = %s, Comment = %s 
            WHERE PromptName = %s
            """
        
            # Execute the update query with the new values and promptname
            self.cursor.execute(sql_update_query, (new_promptstring, new_comment, promptname))
            self.conn.commit()  # Commit the changes
            return "Prompt record updated successfully."
        
        except Error as e:
            self.conn.rollback()  # Rollback the transaction in case of error
            return f"Error occurred while updating the prompt record: {e}"

    # za pretragu promptstrings
    def search_for_string_in_prompt_text(self, search_string):
        """
        Lists all prompt_name and prompt_text where a specific string is part of the prompt_text.

        Parameters:
        - search_string: The string to search for within prompt_text.

        Returns:
        - A list of dictionaries, each containing 'prompt_name' and 'prompt_text' for records matching the search criteria.
        """
        self.cursor.execute('''
        SELECT PromptName, PromptString
        FROM PromptStrings
        WHERE PromptString LIKE %s
        ''', ('%' + search_string + '%',))
        results = self.cursor.fetchall()
    
        # Convert the results into a list of dictionaries for easier use
        records = [{'PromptName': row[0], 'PromptString': row[1]} for row in results]
        return records

    # za pretragu promptstrings po imenu prompta
    def get_prompt_details_by_name(self, promptname):
        """
        Fetches the details of a prompt record identified by PromptName.
    
        :param promptname: The name of the prompt to fetch details for.
        :return: A dictionary with the details of the prompt record, or None if not found.
        """
        query = """
        SELECT PromptName, PromptString, Comment
        FROM PromptStrings
        WHERE PromptName = %s
        """
        try:
            self.cursor.execute(query, (promptname,))
            result = self.cursor.fetchone()
            if result:
                return {"PromptString": result[0], "Comment": result[1]}
            else:
                return None
        except Error as e:
            print(f"Error occurred: {e}")
            return None
        
    # za user i variables i gde god ima samo jedan    
    def update_all_record(self, original_value, new_value, table, column):
        """
        Updates a specific record identified by the original value in a given table and column.
    
        :param original_value: The current value to identify the record to update.
        :param new_value: The new value to set for the specified column.
        :param table: The table to update.
        :param column: The column to update.
        """
        try:
            # Safety check: Validate table and column names
            valid_tables = ['Users', 'PromptVariables', 'PythonFiles']
            valid_columns = ['Username', 'VariableName', 'Filename', 'FilePath']
        
            if table not in valid_tables or column not in valid_columns:
                return "Invalid table or column name."
        
            # Dynamic SQL update statement construction
            sql_update_query = f"""
            UPDATE {table}
            SET {column} = %s
            WHERE {column} = %s
            """
    
            # Execute the update query with the new and original values
            self.cursor.execute(sql_update_query, (new_value, original_value))
            self.conn.commit()  # Commit the changes
            return f"Record updated successfully in {table}."
    
        except Exception as e:
            self.conn.rollback()  # Rollback the transaction in case of error
            return f"Error occurred while updating the record: {e}"

    # za users i variables i gde god ima samo jedan
    def get_prompt_details_for_all(self, value, table, column):
        """
        Fetches the details of a record identified by a value in a specific table and column.

        :param value: The value to fetch details for.
        :param table: The table to fetch from.
        :param column: The column to match the value against.
        :return: A dictionary with the details of the record, or None if not found.
        """
        # Safety check: Validate table and column names
        valid_tables = ['Users', 'PromptVariables', 'PythonFiles']
        valid_columns = ['Username', 'VariableName', 'Filename', 'FilePath']  # Adjust based on your schema
    
        if table not in valid_tables or column not in valid_columns:
            print("Invalid table or column name.")
            return None

        # Dynamic SQL query construction
        query = f"SELECT * FROM {table} WHERE {column} = %s"
    
        try:
            self.cursor.execute(query, (value,))
            result = self.cursor.fetchone()
            if result:
            
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            else:
                return None
        except Error as e:
            print(f"Error occurred: {e}")
            return None
    
    # pomocna funkcija za zatvaranje konekcije
    def close(self):
        """
        Closes the database connection and cursor, if they exist.
        """
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None  # Reset cursor to None to avoid re-closing a closed cursor
        if self.conn is not None and self.conn.is_connected():
            self.conn.close()
            self.conn = None  # Reset connection to None for safety


    # !!!! legacy stari poziv za ostale .py
    def query_sql_record(self, prompt_name):
        """
        Fetches the existing prompt text and comment for a given prompt name.

        Parameters:
        - prompt_name: The name of the prompt.

        Returns:
        - A dictionary with 'prompt_text' and 'comment' if record exists, else None.
        """
        self.cursor.execute('''
        SELECT prompt_text, comment FROM prompts
        WHERE prompt_name = %s
        ''', (prompt_name,))
        result = self.cursor.fetchone()
        if result:
            return {'prompt_text': result[0], 'comment': result[1]}
        else:
            return None

    # privremeno za PythonFiles
    def get_file_path_by_name(self, filename):
        """
        Fetches the FilePath for a given Filename from the PythonFiles table.

        :param filename: The name of the file to fetch the path for.
        :return: The FilePath of the file if found, otherwise None.
        """
        query = "SELECT FilePath FROM PythonFiles WHERE Filename = %s"
        try:
            self.cursor.execute(query, (filename,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    # privremeno za PythonFiles
    def update_filename_and_path(self, original_filename, new_filename, new_file_path):
        """
        Updates the Filename and FilePath in the PythonFiles table for a given original Filename.

        :param original_filename: The original name of the file to update.
        :param new_filename: The new name for the file.
        :param new_file_path: The new path for the file.
        :return: A success message or an error message.
        """
        query = "UPDATE PythonFiles SET Filename = %s, FilePath = %s WHERE Filename = %s"
        try:
            self.cursor.execute(query, (new_filename, new_file_path, original_filename))
            self.conn.commit()
            return "File record updated successfully."
        except Exception as e:
            self.conn.rollback()
            print(f"Error occurred: {e}")
            return None

    def add_relationship_record(self, prompt_id, user_id, variable_id, file_id):
        query = """
        INSERT INTO CentralRelationshipTable (PromptID, UserID, VariableID, FileID)
        VALUES (%s, %s, %s, %s);
        """
        try:
            self.cursor.execute(query, (prompt_id, user_id, variable_id, file_id))
            self.conn.commit()
            return  f"Uspesno dodat {self.cursor.rowcount} slog" # Return the ID of the newly inserted record
        except mysql.connector.Error as e:
            self.conn.rollback()  # Roll back the transaction on error
            return f"Error in add_relationship_record: {e}"

    def update_relationship_record(self, record_id, prompt_id=None, user_id=None, variable_id=None, file_id=None):
        updates = []
        params = []

        if prompt_id:
            updates.append("PromptID = %s")
            params.append(prompt_id)
        if user_id:
            updates.append("UserID = %s")
            params.append(user_id)
        if variable_id:
            updates.append("VariableID = %s")
            params.append(variable_id)
        if file_id:
            updates.append("FileID = %s")
            params.append(file_id)

        if not updates:
            return "No updates provided."

        query = f"UPDATE CentralRelationshipTable SET {', '.join(updates)} WHERE ID = %s;"
        params.append(record_id)

        try:
            self.cursor.execute(query, tuple(params))
            self.conn.commit()
            return "Succesful update relationship record"
        except mysql.connector.Error as e:
            self.conn.rollback()
            return f"Error in update_relationship_record: {e}"

    def delete_record(self, table, condition):
        query = f"DELETE FROM {table} WHERE {condition[0]}"
        try:
            # Directly using condition[1] which is expected to be a list or tuple of values
            self.cursor.execute(query, condition[1])
            self.conn.commit()
            return f"Record deleted"
        except Exception as e:
            self.conn.rollback()
            return f"Error in delete_record: {e}"

    def get_record_by_name(self, table, name_column, value):
        """
        Fetches the entire record from a specified table based on a column name and value.

        :param table: The table to search in.
        :param name_column: The column name to match the value against.
        :param value: The value to search for.
        :return: A dictionary with the record data or None if no record is found.
        """
        query = f"SELECT * FROM {table} WHERE {name_column} = %s"
        try:
            if self.conn is None or not self.conn.is_connected():
                self.__enter__()
            self.cursor.execute(query, (value,))
            result = self.cursor.fetchone()
            if result:
                # Constructing a dictionary from the column names and values
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            else:
                return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
        
    def get_relationships_by_user_id(self, user_id):
        """
        Fetches relationship records for a given user ID.
        
        Parameters:
        - user_id: The ID of the user for whom to fetch relationship records.
        
        Returns:
        - A list of dictionaries containing relationship details.
        """
        relationships = []
        query = """
        SELECT crt.ID, ps.PromptName, u.Username, pv.VariableName, pf.Filename
        FROM CentralRelationshipTable crt
        JOIN PromptStrings ps ON crt.PromptID = ps.PromptID
        JOIN Users u ON crt.UserID = u.UserID
        JOIN PromptVariables pv ON crt.VariableID = pv.VariableID
        JOIN PythonFiles pf ON crt.FileID = pf.FileID
        WHERE crt.UserID = %s
        """
        try:
            # Execute the query with user_id as the parameter
            self.cursor.execute(query, (user_id,))
            records = self.cursor.fetchall()
            
            if records:
                for record in records:
                    relationship = {
                        'ID': record[0],
                        'PromptName': record[1],
                        'Username': record[2],
                        'VariableName': record[3],
                        'Filename': record[4]
                    }
                    relationships.append(relationship)
        except Exception as e:
            # Handle the error appropriately within your application context
            # For example, log the error message
            return False
        
        return relationships
    
    def fetch_relationship_data(self, prompt_id=None):
        # Use self.cursor to execute your query, assuming your class manages a cursor attribute
        query = """
        SELECT crt.ID, ps.PromptName, u.Username, pv.VariableName, pf.Filename
        FROM CentralRelationshipTable crt
        JOIN PromptStrings ps ON crt.PromptID = ps.PromptID
        JOIN Users u ON crt.UserID = u.UserID
        JOIN PromptVariables pv ON crt.VariableID = pv.VariableID
        JOIN PythonFiles pf ON crt.FileID = pf.FileID
        """
        
        # If a prompt_id is provided, append a WHERE clause to filter by that ID
        if prompt_id is not None:
            query += " WHERE crt.PromptID = %s"
            self.cursor.execute(query, (prompt_id,))
        else:
            self.cursor.execute(query)
        
        # Fetch all records
        records = self.cursor.fetchall()
        return records
        

# in myfunc.prompts.py
class ConversationDatabase:
    """
    A class to interact with a MySQL database for storing and retrieving conversation data.
    """
    
    def __init__(self, host=None, user=None, password=None, database=None):
        """
        Initializes the connection details for the database, with the option to use environment variables as defaults.
        """
        self.host = host if host is not None else os.getenv('DB_HOST')
        self.user = user if user is not None else os.getenv('DB_USER')
        self.password = password if password is not None else os.getenv('DB_PASSWORD')
        self.database = database if database is not None else os.getenv('DB_NAME')
        self.conn = None
        self.cursor = None

    def __enter__(self):
        """
        Establishes the database connection and returns the instance itself when entering the context.
        """
        self.conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the database connection and cursor when exiting the context.
        Handles any exceptions that occurred within the context.
        """
        if self.cursor is not None:
            self.cursor.close()
        if self.conn is not None:
            self.conn.close()
        # Handle exception if needed, can log or re-raise exceptions based on requirements
        if exc_type or exc_val or exc_tb:
            # Optionally log or handle exception
            pass
    
    
    def create_sql_table(self):
        """
        Creates a table for storing conversations if it doesn't already exist.
        """
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            app_name VARCHAR(255) NOT NULL,
            user_name VARCHAR(255) NOT NULL,
            thread_id VARCHAR(255) NOT NULL,
            conversation LONGTEXT NOT NULL
        )
        ''')
        # print("Table created if new.")
    
    def add_sql_record(self, app_name, user_name, thread_id, conversation):
        """
        Adds a new record to the conversations table.
        
        Parameters:
        - app_name: The name of the application.
        - user_name: The name of the user.
        - thread_id: The thread identifier.
        - conversation: The conversation data as a list of dictionaries.
        """
        conversation_json = json.dumps(conversation)
        self.cursor.execute('''
        INSERT INTO conversations (app_name, user_name, thread_id, conversation) 
        VALUES (%s, %s, %s, %s)
        ''', (app_name, user_name, thread_id, conversation_json))
        self.conn.commit()
        # print("New record added.")
    
    def query_sql_record(self, app_name, user_name, thread_id):
        """
        Modified to return the conversation record.
        """
        self.cursor.execute('''
        SELECT conversation FROM conversations 
        WHERE app_name = %s AND user_name = %s AND thread_id = %s
        ''', (app_name, user_name, thread_id))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        else:
            return None
    
    def delete_sql_record(self, app_name, user_name, thread_id):
        """
        Deletes a conversation record based on app name, user name, and thread id.
        
        Parameters:
        - app_name: The name of the application.
        - user_name: The name of the user.
        - thread_id: The thread identifier.
        """
        delete_sql = '''
        DELETE FROM conversations
        WHERE app_name = %s AND user_name = %s AND thread_id = %s
        '''
        self.cursor.execute(delete_sql, (app_name, user_name, thread_id))
        self.conn.commit()
        # print("Conversation thread deleted.")
    
    def list_threads(self, app_name, user_name):
        """
        Lists all thread IDs for a given app name and user name.
    
        Parameters:
        - app_name: The name of the application.
        - user_name: The name of the user.

        Returns:
        - A list of thread IDs associated with the given app name and user name.
        """
        self.cursor.execute('''
        SELECT DISTINCT thread_id FROM conversations
        WHERE app_name = %s AND user_name = %s
        ''', (app_name, user_name))
        threads = self.cursor.fetchall()
        return [thread[0] for thread in threads]  # Adjust based on your schema if needed
  
    def update_sql_record(self, app_name, user_name, thread_id, new_conversation):
        """
        Replaces the existing conversation data with new conversation data for a specific record in the conversations table.

        Parameters:
        - app_name: The name of the application.
        - user_name: The name of the user.
        - thread_id: The thread identifier.
        - new_conversation: The new conversation data to replace as a list of dictionaries.
        """

        # Convert the new conversation to JSON format
        new_conversation_json = json.dumps(new_conversation)

        # Update the record with the new conversation
        self.cursor.execute('''
        UPDATE conversations
        SET conversation = %s
        WHERE app_name = %s AND user_name = %s AND thread_id = %s
        ''', (new_conversation_json, app_name, user_name, thread_id))
        self.conn.commit()
        # print("Record updated with new conversation.")

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()

    # Naredne dve metode su vezane isključivo za upisivanje potrosnje tokena u mysql tabelu
    def add_token_record(self, app_id, model_name, embedding_tokens, complete_prompt, full_response, messages):
        """
        Adds a new record to the database with the provided details.
        """
        mem_correction = 2*len(messages) + 1
        prompt_tokens = self.num_tokens_from_messages(complete_prompt, model_name)
        completion_tokens = self.num_tokens_from_messages(messages, model_name) - mem_correction

        # completion
        tiktoken_completion_tokens = self.num_tokens_from_messages(full_response, model_name)
        # prompt
        tiktoken_total_prompt_tokens= prompt_tokens + completion_tokens

        sql = """
        INSERT INTO chatbot_token_log (app_id, embedding_tokens, prompt_tokens, completion_tokens, model_name)
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (app_id, embedding_tokens, tiktoken_total_prompt_tokens, tiktoken_completion_tokens, model_name)
        self.cursor.execute(sql, values)
        self.conn.commit()
    
    def num_tokens_from_messages(self, messages, model):
        """
        Return the number of tokens used by a list of messages.
        """
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
        num_tokens = 0

        if type(messages) == list:
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        elif type(messages) == str:
            num_tokens += len(encoding.encode(messages))
        return num_tokens



# in myfunc.prompts.py
class SQLSearchTool:
    """
    A tool to search an SQL database using natural language queries.
    This class uses the LangChain library to create an SQL agent that
    interprets natural language and executes corresponding SQL queries.
    """

    def __init__(self, db_uri=None):
        """
        Initialize the SQLSearchTool with a database URI.

        :param db_uri: The database URI. If None, it reads from the DB_URI environment variable.
        """

        if db_uri is None:
            db_uri = os.getenv("DB_URI")
        self.db = SQLDatabase.from_uri(db_uri)

        llm = ChatOpenAI(model=work_vars["names"]["openai_model"], temperature=0)
        toolkit = SQLDatabaseToolkit(
            db=self.db, llm=llm
        )

        self.agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            
        )

    def search(self, query, queries=10):
        """
        Execute a search using a natural language query.

        :param query: The natural language query.
        :param queries: The number of results to return (default 10).
        :return: The response from the agent executor.
        """
        with PromptDatabase() as db:
            prompt_map = db.get_prompts_by_names(["sql_search_method"],[os.getenv("SQL_SEARCH_METHOD")])
            sql_search_method = prompt_map.get('sql_search_method', 'You are helpful assistant that always writes in Serbian.').format(query=query, queries=queries)
        try:
            response = self.agent_executor.invoke({sql_search_method})["output"]
        except Exception as e:
            
            response = f"Ne mogu da odgovorim na pitanje, molim vas korigujte zahtev. Opis greske je \n {e}"
        
        return response
