import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import psycopg2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "table_name" not in st.session_state:
    st.session_state.table_name = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "context" not in st.session_state:
    st.session_state.context = ""

# Database and schema functions
def get_schema(table_name, conn):
    cursor = conn.cursor()
    # Use explicit schema to match pgAdmin's context
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{table_name.lower()}'")
    columns = [row[0] for row in cursor.fetchall()]
    cursor.close()
    if not columns:
        raise ValueError(f"No columns found for table {table_name}. Please ensure the table exists in the 'public' schema.")
    schema = f"- {table_name}({', '.join(columns)})"
    return schema

def init_db(df, table_name, conn):
    if not table_name:
        raise ValueError("Table name cannot be empty.")
    cursor = conn.cursor()
    # Check if table exists before dropping
    cursor.execute(f"SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = '{table_name.lower()}')")
    table_exists = cursor.fetchone()[0]
    if table_exists:
        cursor.execute(f"DROP TABLE {table_name}")
    # Infer data types for columns
    columns = df.columns.tolist()
    column_defs = []
    for col in columns:
        if pd.api.types.is_integer_dtype(df[col]):
            column_defs.append(f"{col} INTEGER")
        elif pd.api.types.is_float_dtype(df[col]):
            column_defs.append(f"{col} FLOAT")
        else:
            column_defs.append(f"{col} TEXT")
    create_table_query = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
    cursor.execute(create_table_query)
    # Insert data
    for _, row in df.iterrows():
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            cursor.execute(insert_query, tuple(row))
        except Exception as e:
            st.error(f"Error inserting row: {e}, Row data: {row}")
            raise
    conn.commit()
    cursor.close()
    return f"Database initialized and populated in PostgreSQL for table {table_name}"


# Setup LangChain
API_key = os.getenv("GROQ_API_KEY")
if API_key is None:
    raise RuntimeError("Set your GROQ_API_KEY environment variable.")
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=2000, groq_api_key=API_key)
template = """
Answer the question based on the following context:
{context}
You are an expert SQL assistant. Your job is to translate user questions about any database into valid, safe, and syntactically correct SQLite SQL queries.

**Instructions:**
- Output only the SQL query. Do not include explanations, formatting, or extra text.
- Use only the tables and columns provided below.
- Infer relationships from foreign keys or column names if needed.
- If the question is ambiguous, make reasonable assumptions using the schema context.
- For pie chart requests (e.g., containing 'pie chart', 'proportion', or 'introvert vs extrovert'), generate a query that returns exactly two columns: one for labels (e.g., 'Introvert', 'Extrovert') and one for numeric values (e.g., percentages or counts) using CAST(SUM(CASE WHEN ...) AS REAL) * 100 / COUNT(*) for percentages or COUNT(*) for counts.
- Never include comments or explanations.

**Database Schema:**
{DB_schema}

**Examples:**
Q: List all records from the "students" table.
A: SELECT * FROM students;

Q: Show the average salary by department.
A: SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department;

Q: Find all orders placed in 2024.
A: SELECT * FROM orders WHERE strftime('%Y', order_date) = '2024';

Q: Show the proportion of Introverts vs. Extroverts.
A: SELECT CAST(SUM(CASE WHEN Personality = 'Introvert' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS introvert_percentage, CAST(SUM(CASE WHEN Personality = 'Extrovert' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS extrovert_percentage FROM personality_dataset;

Q: Draw a pie chart of introvert vs extrovert.
A: SELECT Personality, COUNT(*) AS count FROM personality_dataset GROUP BY Personality;

Q: List products with stock less than 10.
A: SELECT * FROM products WHERE stock < 10;

Now, given a user question, output only the SQL query.

User Question: {question}
SQL:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Query execution and visualization functions
def execute_sql(sql, table_name,conn):
    cur = conn.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        if results:
            df = pd.DataFrame(results, columns=columns)
            if df.empty:
                return "Query executed successfully, but no results returned."
            else:
                st.dataframe(df)
                return f"```sql\n{sql}\n```\n{df.to_markdown(index=False)}"
        else:
            return "No results found."
    except Exception as e:
        return f"Error executing SQL: {e}"
    finally:
        conn.close()

def plot_graph(table_name, sql, chart_type=None,conn=None):
    df = pd.read_sql_query(sql, conn)
    conn.close()
    if df.empty:
        return "No data to plot."
    # Allow up to 2 columns for plotting, use first two if more are present
    if len(df.columns) > 2:
        st.warning(f"Query returned {len(df.columns)} columns. Using first two for plotting: {df.columns[0]} and {df.columns[1]}")
        df = df[[df.columns[0], df.columns[1]]]
    if chart_type is None:
        if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            chart_type = 'bar'
        elif len(df.columns) == 1:
            chart_type = 'hist'
        else:
            chart_type = 'bar'
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if chart_type == "pie":
        if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
        elif len(df.columns) == 2 and df.iloc[:, 1].dtype in ['int64', 'float64']:  # Handle count-based data
            total = df[df.columns[1]].sum()
            percentages = [x / total * 100 for x in df[df.columns[1]]]
            ax.pie(percentages, labels=df[df.columns[0]], autopct='%1.1f%%')
        else:
            return f"Pie chart requires two columns with at least one numeric value. Got: {df.columns.tolist()}"
    elif chart_type == "bar":
        if len(df.columns) >= 2:
            ax.bar(df[df.columns[0]], df[df.columns[1]], color='skyblue')
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df[df.columns[1]])
        else:
            df.plot(kind='bar', ax=ax)
    elif chart_type == "line":
        if len(df.columns) >= 2:
            ax.plot(df[df.columns[0]], df[df.columns[1]], marker='o')
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df[df.columns[1]])
        else:
            df.plot(kind="line", ax=ax)
    elif chart_type == "hist":
        df.hist(ax=ax)
    else:
        return f"Chart type '{chart_type}' not recognized. Showing table instead.\n{df.to_markdown(index=False)}"
    
    ax.set_title(" | ".join(df.columns))
    plt.tight_layout()
    return fig

# Streamlit UI
st.set_page_config(page_title="SQL Chatbot with Visualization", layout="wide")
st.title("Data Analysis Chatbot with Visualization")
st.write("Upload a CSV file to start interacting with your data using SQL queries and visualizations.")
file_upload = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")

# Database connection
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise RuntimeError("Set the DATABASE_URL environment variable with your PostgreSQL connection string.")
conn = psycopg2.connect(db_url)

# Handle file upload
if file_upload and "db_initialized" not in st.session_state:
    try:
        df = pd.read_csv(file_upload)
        base_name = os.path.basename(file_upload.name)
        table_name = os.path.splitext(base_name)[0]
        st.session_state.table_name = table_name
        init_msg = init_db(df, table_name,conn)
        st.session_state.db_initialized = True # Mark database as initialized
        st.session_state.schema = get_schema(table_name,conn)
        st.session_state.context = ""  # Reset context on new file upload
        st.success(init_msg)
        st.subheader("Data Preview")
        st.dataframe(df.head(5))
        st.subheader("Database Schema")
        st.code(st.session_state.schema)
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Chat interface
if st.session_state.table_name:
    st.subheader("Chat with the Data Analysis Bot")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("is_plot"):
                st.pyplot(message["content"])
            else:
                st.markdown(message["content"])
    
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            # Get SQL query from LangChain
            llm_output = chain.invoke({
                "DB_schema": st.session_state.schema,
                "context": st.session_state.context,
                "question": user_input
            })
            sql = llm_output.content.strip()

            # Determine if visualization is requested
            chart_type = None
            if any(word in user_input.lower() for word in ["plot", "graph", "chart", "visualize", "visualization", "line", "bar", "pie", "histogram"]):
                if "bar" in user_input.lower():
                    chart_type = "bar"
                if "line" in user_input.lower():
                    chart_type = "line"
                if "pie" in user_input.lower():
                    chart_type = "pie"
                if "histogram" in user_input.lower() or "hist" in user_input.lower():
                    chart_type = "hist"

            # Execute query or plot
            if chart_type:
                response = plot_graph(st.session_state.table_name, sql, chart_type,conn)
                is_plot = not isinstance(response, str)  # True if response is a figure
                if isinstance(response, str):
                    response = f"Visualization error: {response}"
            else:
                if sql.lower().startswith("select"):
                    response = execute_sql(sql, st.session_state.table_name,conn)
                    is_plot = False
                else:
                    response = "Sorry, I can only answer questions, not modify data."
                    is_plot = False

            # Update context
            new_context = st.session_state.context + f"\nUser: {user_input}\nAI: {sql}"
            st.session_state.context = new_context

            # Append and display assistant response
            st.session_state.messages.append({"role": "assistant", "content": response, "is_plot": is_plot})
            with st.chat_message("assistant"):
                if is_plot:
                    st.pyplot(response)
                else:
                    st.markdown(response)
            st.rerun()
        except Exception as e:
            st.error(f"Error processing your query: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "is_plot": False})
            st.rerun()
else:
    st.info("Please upload a CSV file to proceed.")