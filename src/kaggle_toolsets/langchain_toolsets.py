from kaggle_toolsets import run_command

def install_langchain():
    """
    Install the langchain package using pip.
    """
    run_command("pip install langchain -q")

def install_langgraph():
    """
    Install the langgraph package using pip.
    """
    run_command("pip install langgraph -q")

def install_langgraph_openai():
    """
    Install the langgraph-openai package using pip.
    """
    run_command("pip install langchain-openai -q")