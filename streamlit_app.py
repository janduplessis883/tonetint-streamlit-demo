import streamlit as st
from tonetint.sentiment_visualizer import ToneTint
import nltk
from io import BytesIO

# Download the required NLTK resource
nltk.data.path.append('nltk_files/')  # Make sure the path exists and is writable
nltk.download('punkt_tab', download_dir='nltk_files/')
nltk.download('punkt', download_dir='nltk_files/')



st.title("ToneTint - Sentiment Visualizer")

# Select model from dropdown
selected_model = st.selectbox(
    "Pick a Text Classification Model",
    options=[
        "finiteautomata/bertweet-base-sentiment-analysis",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
)

check_box = st.toggle("Load demo text")

if check_box:
    initial_text = """It was a crisp, sunny morning in London, the kind of day that makes the city sparkle. I had plans to meet a friend for brunch at Ollie’s Restaurant, tucked away in a charming corner of town. The weather was perfect as I strolled through the streets, feeling excited about catching up. London is always a buzz of activity, and today was no exception. The sky was clear, and the sunshine bounced off the old architecture, making everything feel more alive. The promise of bottomless brunch at Ollie’s set the tone for a great start to the day.

    However, the journey there wasn’t without its frustrations. I absolutely hate traffic in London. It’s relentless—no matter how much you plan ahead, you’re bound to hit a bottleneck somewhere. Today was no different. A 20-minute drive turned into nearly an hour, crawling through the city’s narrow streets, all because of roadworks and endless buses clogging up the lanes. At moments like these, it’s hard not to think about how much better London could be without the constant gridlock. I was relieved to finally park and make my way to Ollie’s, albeit a bit later than planned.

    Once inside the restaurant, though, all the stress of the journey melted away. Ollie’s had a warm and inviting atmosphere, with an upbeat vibe perfect for a weekend brunch. My friend was already there, and we immediately launched into conversation as if no time had passed. The bottomless brunch was spectacular—delicious food paired with an endless flow of drinks. We lingered over avocado toast, eggs benedict, and mimosas, soaking in the lively buzz of the restaurant. Every time our glasses were empty, they were magically refilled. It was one of those meals where time seemed to slow down, and all that mattered was good food, good drinks, and great company.

    After brunch, we decided to head over to the King’s Road to explore. It’s one of my favorite areas of London, with its mix of high-end boutiques, quirky shops, and picturesque streets. As we walked, we ducked into a few stores, trying on clothes we couldn’t afford and laughing at the outrageous price tags. There’s something special about King’s Road—the history and charm blend perfectly with its modern, trendy vibe. We even found a little café to grab some coffee and rest our feet after all the walking.

    As beautiful as the day was, though, there were a few moments where the hustle of London got to me. The crowds on King’s Road can be overwhelming at times, especially with tourists moving at a snail’s pace and making it impossible to navigate the sidewalks. I love London, but sometimes I wish it was less chaotic, especially on days when all you want is a relaxing stroll. Still, despite the hiccups, it was one of those days that reminds you why London, with all its quirks, is such an incredible place to live."""
    text_area = st.text_area("", value=initial_text, height=250)

else:
    text_area = st.text_area("", placeholder="Paste text here", height=250)


# Slider for chunk size
chunk_size = st.slider("Select chunk size", min_value=6, max_value=18, value=8)

# Color pickers for sentiment colors
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    red_picker = st.color_picker("NEG Color", value="#e8a56c")
with c2:
    yellow_picker = st.color_picker("NEUT Color", value="#f0e8d2")
with c3:
    green_picker = st.color_picker("POS Color", value="#aec867")

# Button to trigger text analysis
button = st.button("Analyze Text")

def init_model():
    # Initialize ToneTint with selected model and colors
    colors = {"NEG": red_picker, "NEU": yellow_picker, "POS": green_picker}
    visualizer = ToneTint(model_name=selected_model, chunk_size=chunk_size, colors=colors)
    return visualizer

# When button is clicked, analyze the text and display the result
# When button is clicked, analyze the text and display the result
if button:
    with st.spinner("Analyzing text, please wait..."):
        visualizer = init_model()  # Initialize the ToneTint model
        html_content = visualizer.visualize(text_area)  # Generate HTML from analyzed text

    with st.container(border=True):
        st.markdown(html_content, unsafe_allow_html=True)  # Display the HTML content

    # Convert HTML content to bytes
    html_bytes = html_content.encode('utf-8')

    # Download HTML
    st.download_button(
        label="Download as HTML",
        data=html_bytes,
        file_name="analyzed_text.html",
        mime="text/html"
    )
