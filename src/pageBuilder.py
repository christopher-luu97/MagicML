from abc import ABC, abstractmethod
import streamlit as st

class PageBuilderInterface(ABC):
    """
    Interface for building Streamlit pages
    """
    
    @abstractmethod
    def app():
        """
        Executor for generating what is needed
        """
        pass