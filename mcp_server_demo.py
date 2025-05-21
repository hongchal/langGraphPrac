from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("DEMO")

@mcp.tool()
def add(a: int, b: int) -> int:
    '''add two numbers together'''
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    '''subtract two numbers'''
    return a - b

if __name__ == "__main__":
    mcp.run(transport='stdio')