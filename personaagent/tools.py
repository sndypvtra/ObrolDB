import sqlite3
from contextlib import contextmanager
from typing import Any, List
import pandas as pd
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from personaagent.config import Config
from personaagent.logging import log, log_panel


def get_available_tools() -> List[BaseTool]:
    return [
        list_tables,
        get_columns,
        count_rows,
        describe_table,
        sample_table,
        get_primary_keys,
        get_foreign_keys,
        execute_sql,
    ]

def call_tool(tool_call: ToolCall) -> Any:
    """
    Invoke the specified tool with the provided arguments and wrap the result.

    Args:
        tool_call: A ToolCall dict containing name and args for the tool.

    Returns:
        A ToolMessage containing the tool's output and its call ID.
    """
    tools_by_name = {t.name: t for t in get_available_tools()}
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])
    id_key = next((k for k in tool_call.keys() if "id" in k.lower()), None)
    call_id = str(tool_call.get(id_key, ""))
    return ToolMessage(content=result, tool_call_id=call_id)

@contextmanager
def with_sql_cursor(readonly: bool = True):
    """
    Provide a SQLite cursor in a context manager.

    Args:
        readonly: If False, commit changes; otherwise do not commit.

    Yields:
        A sqlite3.Cursor connected to the configured database.
    """
    conn = sqlite3.connect(Config.Path.DATABASE_PATH)
    cur = conn.cursor()
    try:
        yield cur
        if not readonly:
            conn.commit()
    except:
        if not readonly:
            conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

@tool(parse_docstring=True)
def list_tables(reasoning: str = "") -> str:
    """
    List all user-created tables in the database.

    Args:
        reasoning: Optional explanation for why tables are listed.

    Returns:
        A string containing a formatted list of table names with insights.
    """
    log_panel(title="List Tables", content=f"Reasoning: {reasoning}")
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row[0] for row in cursor.fetchall()]
        
        # Format sebagai daftar Markdown
        table_list = "\n".join([f"- {table}" for table in tables])
        insight = f"**Insight:** Database berisi {len(tables)} tabel yang dapat diakses."
        
        return f"**Daftar Tabel:**\n{table_list}\n\n{insight}"
    except Exception as e:
        log(f"[red]Error listing tables: {e}[/red]")
        return "Error listing tables."

@tool(parse_docstring=True)
def get_columns(table_name: str, reasoning: str = "") -> str:
    """
    Get comma-separated column names for a specific table.

    Args:
        table_name: Exact name of the table (case-sensitive).
        reasoning: Optional explanation for why column info is needed.

    Returns:
        Comma-separated list of column names.
    """
    log_panel(title="Get Columns", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f'PRAGMA table_info("{safe}");')
            cols = [row[1] for row in cursor.fetchall()]
        return ", ".join(cols)
    except Exception as e:
        log(f"[red]Error getting columns: {e}[/red]")
        return "Error getting columns."

@tool(parse_docstring=True)
def count_rows(table_name: str, reasoning: str = "") -> str:
    """
    Count the number of rows in a table.

    Args:
        table_name: Exact name of the table.
        reasoning: Optional rationale for needing the row count.

    Returns:
        Total row count as a string.
    """
    log_panel(title="Count Rows", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{safe}";')
            count = cursor.fetchone()[0]
        return str(count)
    except Exception as e:
        log(f"[red]Error counting rows: {e}[/red]")
        return "Error counting rows."

@tool(parse_docstring=True)
def describe_table(table_name: str, reasoning: str = "") -> str:
    """
    Describe table schema (columns, types, constraints).

    Args:
        table_name: Exact name of the table.
        reasoning: Optional reason for needing schema details.

    Returns:
        One row per column from PRAGMA table_info as tuples.
    """
    log_panel(title="Describe Table", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f'PRAGMA table_info("{safe}");')
            rows = cursor.fetchall()
        return "\n".join(str(row) for row in rows)
    except Exception as e:
        log(f"[red]Error describing table: {e}[/red]")
        return "Error describing table."

@tool(parse_docstring=True)
def sample_table(reasoning: str = "", table_name: str = "", row_sample_size: int = 10) -> str:
    """
    Retrieve a sample of rows from a table and return as a formatted table with insights.

    Args:
        reasoning: Optional explanation for sampling data.
        table_name: Exact name of the table.
        row_sample_size: Number of rows to retrieve.

    Returns:
        A string containing a table representation and insights.
    """
    log_panel(title="Sample Table", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f'SELECT * FROM "{safe}" LIMIT {row_sample_size};')
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(rows, columns=columns)
        table_str = df.to_markdown(index=False)
        
        insight = f"**Insight:** Menampilkan {len(df)} baris sampel dari tabel {table_name}. "
        insight += "Data ini memberikan gambaran awal tentang isi tabel."
        
        return f"{table_str}\n\n{insight}"
    except Exception as e:
        log(f"[red]Error sampling table: {e}[/red]")
        return "Error sampling table."

@tool(parse_docstring=True)
def get_primary_keys(table_name: str, reasoning: str = "") -> str:
    """
    Get primary key columns for a table.

    Args:
        table_name: Exact name of the table.
        reasoning: Optional explanation for needing PK info.

    Returns:
        Comma-separated primary key column names.
    """
    log_panel(title="Get Primary Keys", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    with with_sql_cursor() as cursor:
        cursor.execute(f'PRAGMA table_info("{safe}");')
        rows = cursor.fetchall()
    pks = [col[1] for col in rows if col[5] > 0 or col[1].endswith("ID")]
    return ", ".join(pks)

@tool(parse_docstring=True)
def get_foreign_keys(table_name: str, reasoning: str = "") -> str:
    """
    Get foreign key constraints for a table.

    Args:
        table_name: Exact name of the table.
        reasoning: Optional explanation for needing FK info.

    Returns:
        One constraint per line in the format 'from_col → to_table.to_col'.
    """
    log_panel(title="Get Foreign Keys", content=f"Reasoning: {reasoning}")
    safe = table_name.replace('"', '""')
    with with_sql_cursor() as cursor:
        cursor.execute(f'PRAGMA foreign_key_list("{safe}");')
        rows = cursor.fetchall()
    lines = [f"{r[3]} → {r[2]}.{r[4]}" for r in rows]
    return "\n".join(lines)

@tool(parse_docstring=True)
def execute_sql(reasoning: str = "", sql_query: str = "") -> str:
    """
    Execute any SQL query and return the results as a formatted table with insights.

    Args:
        reasoning: Optional explanation for running query.
        sql_query: The SQL query string to execute (not returned in output).

    Returns:
        A string containing a table representation and insights, without exposing the SQL query.
    """
    log_panel(title="Execute SQL", content=f"Reasoning: {reasoning}")
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            # Ambil nama kolom dari cursor
            columns = [desc[0] for desc in cursor.description]
        
        # Konversi hasil ke DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Buat representasi tabel dalam Markdown
        table_str = df.to_markdown(index=False)
        
        # Generate insight sederhana berdasarkan hasil
        insight = f"**Insight:** Query menghasilkan {len(df)} baris data. "
        if len(df) > 0:
            insight += f"Data menunjukkan informasi relevan dari tabel {columns[0]} dan kolom terkait."
        else:
            insight += "Tidak ada data yang ditemukan untuk permintaan ini."
        
        return f"{table_str}\n\n{insight}"
    except Exception:
        return "An error occurred when executing the query. Please check your request."
