import numpy as np
from typing import Any

class WorldModel:
    """
    this is just a wrapper of the model

    the base model would be a transformer

    action + image_flow + query prompt -> bool

    """

    image_flow: list[np.ndarray] # py 10
    action_summary: str
    query_information: str

    model_interface : Any

    def __init__(self) -> None:
        pass

    def set_action():
        """
        set the currently executed action node
        """
        pass

    def set_image_flow():
        """
        set the image flow of the workspace
        """
        pass

    def set_query():
        """
        set the current query term of the condition node

        with the execution of the process, the query list should grow.
        the horizon of the problem is therefore perhaps pretty much restricted.

        """
        pass


    def rollout():
        pass
        # 
