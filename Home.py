import streamlit as st

def main():
    """
    This acts as the entrypoint to the application
    """
    st.set_page_config(
        page_title="Magic ML"
    )

    st.write("# Welcome to the Magic ML Web App!")
    st.sidebar.success("Select a page above.")

    st.markdown(
        """
        Magic ML is your all-in-one tool to accelerate proof-of-concept development
        within data science!
        - Exploratory data analysis
        - Model building
        **👈 Select your process from the sidebar** to begin!
    """
    )

if __name__ == '__main__':
    main()