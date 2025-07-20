"""
Enhanced Zephyr Dashboard with Chat Capabilities

This module extends the original dashboard with a chat interface
that uses FAISS for semantic search of comments.
"""

# Import warning suppression before anything else
import warnings

# Suppress PyTorch class registration warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*torch.tensor results are registered as constants.*",
)

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
import sys
import os

# Add parent directory to path to import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat.chat_engine import ZephyrChatEngine


# Get absolute path to assets directory
def get_asset_path(asset_name):
    """
    Get the absolute path to an asset in the _assets directory.

    Args:
        asset_name (str): The name of the asset file

    Returns:
        str: Absolute path to the asset
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look in the img subfolder in the same directory as the file
    return os.path.join(module_dir, "img", asset_name)


# Get path to CSS file
def get_css_path():
    """Get the path to the CSS file."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "styles", "styles.css")


# Zephyr-themed styling - load external CSS
st.set_page_config(page_title="Zephyr Analytics Dashboard", layout="wide")

# Load CSS from external file
with open(get_css_path(), "r") as f:
    css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_sentiment_data(filename="sentiment_results.json"):
    """Load the sentiment analysis results from JSON file"""
    try:
        # Get the module directory and parent directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(module_dir)
        
        # Path to the sentiment file in ../data/
        data_dir = os.path.join(parent_dir, "data")
        file_path = os.path.join(data_dir, filename)
        
        # Load the sentiment data
        with open(file_path, "r") as f:
            return json.load(f)
            
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return []


def calculate_product_sentiment(data):
    """Calculate average sentiment score for each product"""
    df = pd.DataFrame(data)
    df["score"] = df["sentiment"].apply(lambda x: x["score"])
    return df.groupby("product")["score"].mean().to_dict()


# Initialize chat engine (cached to avoid rebuilding on every interaction)
@st.cache_resource
def get_chat_engine():
    """Get the chat engine with FAISS vector database"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentiment_data_path = os.path.join(root_dir, "data", "sentiment_results.json")
    faiss_path = os.path.join(root_dir, "data", "faiss_index.pkl")

    engine = ZephyrChatEngine(sentiment_data_path)
    success = False

    # Try to load the index
    try:
        engine.load_index(faiss_path)
        success = True
    except Exception as e:
        st.warning(f"Error with search index: {e}")

    return engine, success


def dashboard_tab():
    """Render the original dashboard tab"""
    # Header - removed subtitle as requested
    st.title("🚀 Zephyr Innovations Sentiment Dashboard")

    # Load data
    data = load_sentiment_data()
    if not data:
        return

    df = pd.DataFrame(data)
    df["score"] = df["sentiment"].apply(lambda x: x["score"])
    df["date"] = pd.to_datetime(df["date"])

    # Create main layout - report on left, charts stacked on right
    report_col, charts_col = st.columns([1, 2])

    # Management report on the left
    with report_col:
        st.header("📊 Zephyr Management Report")

        # Calculate report data
        product_sentiment = calculate_product_sentiment(data)
        total_posts = len(data)
        positive = len(df[df["score"] > 0])
        negative = len(df[df["score"] < 0])
        neutral = total_posts - positive - negative
        best_product = max(product_sentiment, key=product_sentiment.get)
        worst_product = min(product_sentiment, key=product_sentiment.get)

        # Display report
        st.write(
            f"""
        - **Total Posts**: {total_posts}
        - **Positive**: {positive} ({positive/total_posts*100:.1f}%)
        - **Neutral**: {neutral} ({neutral/total_posts*100:.1f}%)
        - **Negative**: {negative} ({negative/total_posts*100:.1f}%)
        - **Top Product**: {best_product} (Score: {product_sentiment[best_product]:.2f})
        - **Needs Work**: {worst_product} (Score: {product_sentiment[worst_product]:.2f})
        """
        )

        # Stack the charts in the right column
    with charts_col:
        # Create a DataFrame for better handling in plotly
        bar_df = pd.DataFrame(
            {
                "Product": list(product_sentiment.keys()),
                "Score": list(product_sentiment.values()),
            }
        )

        # Count comments per product for hover info
        product_counts = df.groupby("product").size().to_dict()
        bar_df["Count"] = bar_df["Product"].map(product_counts)

        # Calculate average for reference line
        avg_sentiment = bar_df["Score"].mean()

        # Create enhanced bar chart
        fig_bar = px.bar(
            bar_df,
            x="Product",
            y="Score",
            color="Score",
            color_continuous_scale=[
                [0, "red"],  # -1.0 (negative)
                [0.4, "red"],  # -0.2
                [0.5, "gray"],  # 0 (neutral)
                [0.6, "green"],  # 0.2
                [1, "green"],  # 1.0 (positive)
            ],
            labels={
                "Product": "Product",
                "Score": "Avg Sentiment Score",
                "Count": "Number of Comments",
            },
            title="Sentiment by Product",
            text="Score",  # Display score on bars
            hover_data=["Count"],  # Show comment count on hover
        )

        # Customize hover template
        fig_bar.update_traces(
            hovertemplate="<b>%{x}</b><br>Sentiment Score: %{y:.2f}<br>Comments: %{customdata[0]}<extra></extra>",
            texttemplate="%{text:.2f}",  # Format the text on bars
            textposition="outside",  # Position text outside bars
        )

        # Add reference line for average
        fig_bar.add_shape(
            type="line",
            x0=-0.5,
            x1=len(product_sentiment) - 0.5,
            y0=avg_sentiment,
            y1=avg_sentiment,
            line=dict(color="blue", width=2, dash="dash"),
        )

        # Add annotation for average line
        fig_bar.add_annotation(
            x=len(product_sentiment) - 1,
            y=avg_sentiment,
            text=f"Avg: {avg_sentiment:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(color="blue"),
        )

        # Enhanced layout
        fig_bar.update_layout(
            yaxis_range=[-1.1, 1.1],  # Slightly expanded to accommodate text
            title_font={"size": 24},
            xaxis_title_font={"size": 16},
            yaxis_title_font={"size": 16},
            yaxis_gridcolor="lightgray",
            yaxis_zeroline=True,
            yaxis_zerolinecolor="black",
            yaxis_zerolinewidth=2,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Continue with the charts column - now showing just the trend chart
    with charts_col:
        # Trend chart below the bar chart
        trend_df = (
            df.groupby(["product", pd.Grouper(key="date", freq="W")])["score"]
            .mean()
            .reset_index()
        )
        fig_trend = px.line(
            trend_df,
            x="date",
            y="score",
            color="product",
            labels={"date": "Date", "score": "Avg Sentiment"},
            title="Sentiment Trends Over Time",
        )
        # Increase title font size for the trend chart
        fig_trend.update_layout(
            yaxis_range=[-1, 1],
            title_font={"size": 24},  # Increase title font size (default is ~18)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Continue with the report column - aligned precisely with the trend chart
    with report_col:
        # Add a centered Zephyr logo between the management report and product table
        col1, col2, col3 = st.columns([1, 2, 1])  # Create columns for centering
        with col2:  # Use middle column to center the smaller image
            try:
                image_path = get_asset_path("zephyr-innovations-dashboard.png")
                if os.path.exists(image_path):
                    st.image(image_path, width=226)
                else:
                    st.info(f"Dashboard logo image not found at: {image_path}")
            except Exception as e:
                st.warning(f"Could not load dashboard logo: {str(e)}")

        # Add Product Table with styling - now aligned with the trend chart
        st.subheader("Zephyr Product Line")

        # Create list of products and their sentiment scores
        product_list = [
            "Anvil Drop Kit",
            "Bird Seed",
            "Giant Magnet",
            "Rocket Skates",
            "TNT Sticks",
        ]

        # Create a DataFrame for the product table
        product_data = []

        # Build data for each product
        for product in product_list:
            # Get product score
            score = product_sentiment.get(product, 0)

            # Determine sentiment icon based on score
            if score > 0.2:
                sentiment_icon = "📈"
            elif score < -0.2:
                sentiment_icon = "📉"
            else:
                sentiment_icon = "📊"

            # Format the score
            score_formatted = f"{sentiment_icon} {score:.2f}"

            # Add to data
            product_data.append(
                {"Product": product, "Sentiment Score": score_formatted}
            )

        # Create DataFrame and reset index to start from 1 instead of 0
        product_df = pd.DataFrame(product_data)
        # Reset index and add 1 to make it start from 1
        product_df.index = range(1, len(product_df) + 1)

        # Display the table using st.table
        st.table(product_df)

        # Add a note or tip that takes up some vertical space
        st.info(
            """
            💡 **Tip**: Our sentiment analyzer correctly understands that explosions 
            and destruction are positive attributes for TNT Sticks!
            """
        )


def chat_tab():
    """Render the chat tab with FAISS-powered comment search"""
    # Initialize session state for chat history if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Place the input area at the very top for immediate visibility
    st.subheader("Ask a question about customer comments")

    # Text area for input (directly below the header)
    query = st.text_area(
        label="Type your question here",
        value="What do customers like about TNT Sticks?",
        height=68,
    )

    # Ask button
    if st.button("Ask", use_container_width=True):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": query})

        # Initialize the chat engine
        try:
            engine, _ = get_chat_engine()

            # Get the k value from session state
            k = st.session_state.get("k_value", 5)

            # Prepare filters
            filters = {}
            if "product_filter" in st.session_state and st.session_state.product_filter:
                filters["product"] = st.session_state.product_filter
            if (
                "sentiment_filter" in st.session_state
                and st.session_state.sentiment_filter
            ):
                filters["sentiment"] = st.session_state.sentiment_filter
            if (
                "commenter_filter" in st.session_state
                and st.session_state.commenter_filter
            ):
                filters["commenter"] = st.session_state.commenter_filter

            if (
                "use_date_filter" in st.session_state
                and st.session_state.use_date_filter
                and "date_range" in st.session_state
            ):
                date_range = st.session_state.date_range
                if len(date_range) == 2:
                    filters["date_range"] = (
                        date_range[0].strftime("%Y-%m-%d"),
                        date_range[1].strftime("%Y-%m-%d"),
                    )

            # Apply empty filters as None
            if not filters:
                filters = None

            # Generate response
            with st.spinner("Searching for relevant comments..."):
                result = engine.chat(query, k, filters)

                # Create AI response message
                ai_message = {
                    "role": "assistant",
                    "content": (
                        result["answer"]
                        if "error" not in result
                        else f"⚠️ {result['answer']}"
                    ),
                    "relevant_comments": result.get("relevant_comments", []),
                }

                # Add AI message to chat history
                st.session_state.chat_messages.append(ai_message)

                # Rerun to update the chat display
                st.rerun()

        except Exception as e:
            st.error(f"Error generating response: {e}")

    # Create columns for chat history and filters
    chat_col, filter_col = st.columns([0.8, 0.2])

    # Show chat history below the input (most recent first)
    with chat_col:
        st.subheader("Conversation (most recent first)")

        # Group messages into Q&A pairs and display in reverse order
        messages = st.session_state.chat_messages.copy()
        qa_pairs = []

        # Create pairs of questions and answers
        i = 0
        while i < len(messages):
            if (
                i + 1 < len(messages)
                and messages[i]["role"] == "user"
                and messages[i + 1]["role"] == "assistant"
            ):
                qa_pairs.append((messages[i], messages[i + 1]))
                i += 2
            else:
                # Handle odd message (usually a standalone question without answer yet)
                if i < len(messages):
                    qa_pairs.append((messages[i], None))
                i += 1

        # Display pairs in reverse order (most recent first)
        for user_msg, ai_msg in reversed(qa_pairs):
            # Create a container for each Q&A pair
            with st.container():
                # Always show the user question
                st.markdown(
                    f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin-bottom: 5px;">{user_msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Show the AI response if it exists
                if ai_msg:
                    st.markdown(
                        f'<div style="background-color: #f1f3f4; padding: 10px; border-radius: 10px; margin-bottom: 20px;">{ai_msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Show relevant comments in expander
                    if "relevant_comments" in ai_msg:
                        with st.expander("Show relevant comments"):
                            for i, comment in enumerate(ai_msg["relevant_comments"]):
                                sentiment_class = {
                                    "POSITIVE": "sentiment-positive",
                                    "NEUTRAL": "sentiment-neutral",
                                    "NEGATIVE": "sentiment-negative",
                                }[comment["sentiment"]["label"]]

                                st.markdown(
                                    f"**{i+1}. {comment['product']} comment by {comment['commenter']}**"
                                )
                                st.write(f"**Comment:** {comment['text']}")
                                st.markdown(
                                    f"**Sentiment:** <span class='{sentiment_class}'>{comment['sentiment']['label']}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.write(f"**Date:** {comment['date']}")
                                st.markdown("---")

    # Show filters in the sidebar
    with filter_col:
        try:
            # Initialize engine for filter options
            engine, index_loaded = get_chat_engine()
            unique_values = engine.get_unique_values()

            with st.expander("Filters", expanded=False):
                # Number of comments slider
                k_value = st.slider(
                    "Number of comments:",
                    3,
                    25,
                    5,
                    key="k_value",
                )

                # Product filter
                st.markdown("<strong>Product:</strong>", unsafe_allow_html=True)
                product_filter = st.multiselect(
                    "Filter by Product:",
                    options=unique_values["products"],
                    default=[],
                    label_visibility="collapsed",
                    key="product_filter",
                )

                # Sentiment filter
                st.markdown("<strong>Sentiment:</strong>", unsafe_allow_html=True)
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment:",
                    options=unique_values["sentiments"],
                    default=[],
                    label_visibility="collapsed",
                    key="sentiment_filter",
                )

                # Commenter filter
                st.markdown("<strong>Commenter:</strong>", unsafe_allow_html=True)
                commenter_filter = st.multiselect(
                    "Filter by Commenter:",
                    options=unique_values["commenters"],
                    default=[],
                    label_visibility="collapsed",
                    key="commenter_filter",
                )

                # Date range filter
                st.markdown("<strong>Date Range:</strong>", unsafe_allow_html=True)
                min_date, max_date = unique_values["date_range"]
                date_range = st.date_input(
                    "Filter by Date Range:",
                    value=(
                        datetime.strptime(min_date, "%Y-%m-%d").date(),
                        datetime.strptime(max_date, "%Y-%m-%d").date(),
                    ),
                    min_value=datetime.strptime(min_date, "%Y-%m-%d").date(),
                    max_value=datetime.strptime(max_date, "%Y-%m-%d").date(),
                    label_visibility="collapsed",
                    key="date_range",
                )
                use_date_filter = st.checkbox(
                    "Apply Date Filter", value=False, key="use_date_filter"
                )

            # Show FAISS index status
            if index_loaded:
                st.success("✅ FAISS index loaded")
                # Add Zephyr logo below the FAISS index message (full width to fill space)
                try:
                    image_path = get_asset_path("zephyr-innovations-logo-chat.png")
                    if os.path.exists(image_path):
                        st.image(
                            image_path,
                            width=None,  # Let Streamlit handle the sizing automatically
                        )
                    else:
                        st.info(f"Logo image not found at: {image_path}")
                        # List files in _assets directory to help troubleshoot
                        assets_dir = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "_assets",
                        )
                        if os.path.exists(assets_dir):
                            available_assets = ", ".join(os.listdir(assets_dir))
                            st.write(f"Available assets: {available_assets}")
                except Exception as e:
                    st.warning(f"Could not display logo: {str(e)}")

        except Exception as e:
            st.error(f"Error initializing filters: {e}")


def about_tab():
    """Render information about the application"""
    st.title("About Zephyr Analytics")

    st.markdown(
        """
    ## Overview
    
    Zephyr Analytics is a sentiment analysis and insight platform for customer comments about Zephyr Innovation products.
    The platform analyzes comments using Gemma 3, a powerful language model, to determine sentiment and enable
    semantic search using vector embeddings.
    
    ## Features
    
    ### Sentiment Dashboard
    - Visual representation of sentiment across products
    - Sentiment trends over time
    - Live sentiment analysis of new comments
    
    ### AI Chat Interface
    - Ask questions about customer comments
    - Semantic search using FAISS vector database
    - Filter by product, sentiment, commenter, or date
    
    ## Technical Details
    
    ### How the Chat Works
    1. Comments are converted to vector embeddings using sentence-transformers
    2. These embeddings are stored in a FAISS index for fast similarity search
    3. When you ask a question, it's converted to a vector and compared to the comment vectors
    4. The most semantically similar comments are retrieved
    5. Gemma 3 is given these comments as context and generates an answer to your question
    
    ### GPU Acceleration
    This implementation uses `faiss-cpu` for vector similarity search. For larger datasets or faster performance, 
    you can switch to GPU acceleration by installing `faiss-gpu` instead. See the documentation for details.
    """
    )

    st.info(
        """
    **Note:** This application uses local models and doesn't send data to external APIs. 
    All processing happens on your machine through the Ollama API.
    """
    )


def main():
    """Main application entry point"""
    # Initialize session state for tab tracking if not exists
    if "tab" not in st.session_state:
        st.session_state.tab = "Dashboard"

    # Apply selected button styling based on current tab
    if st.session_state.tab == "Dashboard":
        st.markdown(
            """
            <style>
            button[data-testid="baseButton-secondary"][aria-label="dash_btn"] {
                color: black !important;
                font-weight: 900 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.tab == "Chat":
        st.markdown(
            """
            <style>
            button[data-testid="baseButton-secondary"][aria-label="chat_btn"] {
                color: black !important;
                font-weight: 900 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.tab == "About":
        st.markdown(
            """
            <style>
            button[data-testid="baseButton-secondary"][aria-label="about_btn"] {
                color: black !important;
                font-weight: 900 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Create a more compact header layout
    header_cols = st.columns([1, 1])

    with header_cols[0]:
        # Add logo-container class to fix logo display
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        # Logo on left side - using different image with 75% larger size
        try:
            image_path = get_asset_path("zephyr-innovations-sm.png")
            if os.path.exists(image_path):
                st.image(
                    image_path, width=175
                )  # Increased from 100 to 175 (75% larger)
            else:
                st.info(f"Header logo image not found at: {image_path}")
        except Exception as e:
            st.warning(f"Could not load header logo: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with header_cols[1]:
        # Container for right-aligned buttons
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; gap: 20px; margin-top: 10px;">
            """,
            unsafe_allow_html=True,
        )

        # Create right-aligned button columns with minimal spacing
        # Use narrower columns to reduce spacing between buttons
        nav_container = st.container()
        with nav_container:
            nav_cols = st.columns([1, 1, 1])

            with nav_cols[0]:
                dash_btn = st.button(
                    "Dashboard", key="dash_btn", use_container_width=True
                )
                if dash_btn:
                    st.session_state.tab = "Dashboard"
                    st.rerun()

            with nav_cols[1]:
                chat_btn = st.button("Chat", key="chat_btn", use_container_width=True)
                if chat_btn:
                    st.session_state.tab = "Chat"
                    st.rerun()

            with nav_cols[2]:
                about_btn = st.button(
                    "About", key="about_btn", use_container_width=True
                )
                if about_btn:
                    st.session_state.tab = "About"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Add a thinner divider below the navbar
    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 10px;'>", unsafe_allow_html=True
    )

    # Get current tab from session state
    tab = st.session_state.tab

    # Render the selected tab
    if tab == "Dashboard":
        dashboard_tab()
    elif tab == "Chat":
        chat_tab()
    elif tab == "About":
        about_tab()


if __name__ == "__main__":
    main()
