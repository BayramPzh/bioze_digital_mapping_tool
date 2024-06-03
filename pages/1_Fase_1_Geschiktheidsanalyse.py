import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st
import numpy as np
from utils.cflp_function import *
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import networkx as nx
from pysal.lib import weights
from pysal.explore import esda
import plotly.figure_factory as ff

# Constants
PADDING = 0
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=6.747489560596507, latitude=52.316862707395394, zoom=8, bearing=0, pitch=0)

# Set page configuration
st.set_page_config(page_title="Suitability Analysis", layout="wide")

# Set markdown
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(4)
        {
            text-align: end;
        } 
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data
def load_gdf(gdf_path):
    return gpd.read_file(gdf_path).set_index('hex9')

# Load dataframes
d_to_farm = load_data('./hex/farm_v2.csv')
d_to_road = load_data('./hex/road_v2.csv')
d_to_industry = load_data('./hex/industry_v2.csv')
d_to_nature = load_data('./hex/nature_v2.csv')
d_to_water = load_data('./hex/water_v2.csv')
d_to_urban = load_data('./hex/urban_v2.csv')
d_to_inlet = load_data('./hex/inlet_v2.csv')
d_to_pm25 = load_data('./csv/Provincie Zuid-Holland Luchtkwaliteit - Samen Meten Dashboard.csv')


# Generate colormap
color_mapping = generate_color_mapping(COLORMAP)

# Fuzzify input variables
@st.cache_data
def fuzzify(df, type="close", colormap_name=color_mapping):
    df_array = np.array(df['value'])
    fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / (df_array.max() - df_array.min())) if type == "close" else np.maximum(0, (df_array - df_array.min()) / (df_array.max() - df_array.min()))
    df['fuzzy'] = fuzzified_array.round(3)
    apply_color_mapping(df, 'fuzzy', color_mapping)
    return df

# Fuzzify dataframes
fuzzy_farm = fuzzify(d_to_farm, type='close')
fuzzy_road = fuzzify(d_to_road, type='close')
fuzzy_industry = fuzzify(d_to_industry, type='close')
fuzzy_nature = fuzzify(d_to_nature, type='far')
fuzzy_water = fuzzify(d_to_water, type='far')
fuzzy_urban = fuzzify(d_to_urban, type='far')
fuzzy_inlet = fuzzify(d_to_inlet, type='close')
fuzzy_pm25 = fuzzify(d_to_pm25, type='close')  # or type='far',

# All arrays
all_arrays = {'Farms': np.array(fuzzy_farm['fuzzy']), 
              'Road infrastructure': np.array(fuzzy_road['fuzzy']),
              'Urban and residential areas': np.array(fuzzy_urban['fuzzy']), 
              'Industrial areas': np.array(fuzzy_industry['fuzzy']), 
              'Nature': np.array(fuzzy_nature['fuzzy']),
              'Water Bodies': np.array(fuzzy_water['fuzzy']),
              'Gas Inlets': np.array(fuzzy_inlet['fuzzy']),
              'Pm25': np.array(fuzzy_pm25['fuzzy'])}

# Create empty layer
def create_empty_layer(d_to_farm):
    df_empty = d_to_farm[['hex9']]
    df_empty['color'] = '[0,0,0,0]'
    return df_empty

# Update empty df
def update_layer(selected_variables, all_arrays, d_to_farm):
    if not selected_variables:
        return create_empty_layer(d_to_farm)
    
    selected_array_list = [all_arrays[key] for key in selected_variables]
    result_array = np.mean(selected_array_list, axis=0)
    hex_df = create_empty_layer(d_to_farm)
    hex_df['fuzzy'] = result_array
    apply_color_mapping(hex_df, 'fuzzy', color_mapping)
    hex_df['fuzzy'] = hex_df['fuzzy'].round(3)
    return hex_df

# Filter potential digester locations
def get_sites(fuzzy_df, w, g, idx):
    if 'fuzzy' in fuzzy_df.columns:
        fuzzy_df = fuzzy_df.set_index('hex9').reindex(idx.index)
        lisa = esda.Moran_Local(fuzzy_df['fuzzy'], w, seed=42)
        HH = fuzzy_df[(lisa.q == 1) & (lisa.p_sim < 0.01)].index.to_list()
        H = g.subgraph(HH)
        subH = list(nx.connected_components(H))
        filter_subH = [component for component in subH if len(component) > 10]
        site_idx = []
        for component in filter_subH:
            subgraph = H.subgraph(component)
            eigenvector_centrality = nx.eigenvector_centrality(subgraph, max_iter=1500)
            max_node_index = max(eigenvector_centrality, key=eigenvector_centrality.get)
            site_idx.append(max_node_index)
        st.session_state.all_loi = fuzzy_df.loc[site_idx].reset_index()
    else:
        return None

# Generate pydeck
@st.cache_data
def generate_pydeck(df, view_state=VIEW_STATE):
    return pdk.Deck(initial_view_state=view_state,
                    layers=[
                        pdk.Layer(
                            "H3HexagonLayer",
                            df,
                            pickable=True,
                            stroked=True,
                            filled=True,
                            extruded=False,
                            opacity=0.6,
                            get_hexagon="hex9",
                            get_fill_color ='color', 
                        ),
                    ],
                    tooltip={"text": "Suitability:" f"{{fuzzy}}"})

# Create variable legend
@st.cache_data
def generate_colormap_legend(label_left='Far', label_right='Near', cmap=plt.get_cmap(COLORMAP)):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.axis('off') 

    ax.text(-10, 0.5, label_left, verticalalignment='center', horizontalalignment='right', fontsize=12)
    ax.text(266, 0.5, label_right, verticalalignment='center', horizontalalignment='left', fontsize=12)

    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    image_png = buffer.getvalue()
    plt.close(fig)
    image_base64 = base64.b64encode(image_png).decode()

    legend_html = f'''
        <div style="width: 100%; height: 300px; overflow: auto; padding: 10px;">
            <img src="data:image/png;base64,{image_base64}" alt="Colorbar" style="max-width: 100%; max-height: 100%; height: auto; width: auto; display: block; margin-left: auto; margin-right: auto;">
        </div>
    '''
    return legend_html

variable_legend_html = generate_colormap_legend(label_left='Least Suitable (0)', label_right='Most Suitable (1)',)

# Get layers
@st.cache_data
def get_layers(hex_df):
    hex_fuzzy = pdk.Layer(
        "H3HexagonLayer",
        hex_df.reset_index(),
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        opacity=0.6,
        get_hexagon="hex9",
        get_fill_color='color', 
    )

    layers = [hex_fuzzy]
    return layers

# Plot result
def plot_result(fig):
    if fig is not None:
        st.plotly_chart(fig, theme="streamlit")

# Initialize session state
def initialize_session_state(idx):
    if 'all_loi' not in st.session_state:
        st.session_state.all_loi = pd.DataFrame()
    if 'loi' not in st.session_state:
        st.session_state.loi = pd.DataFrame()
    if 'fig' not in st.session_state:
        st.session_state.fig = None
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, use_index=True)
    if 'g' not in st.session_state:
        st.session_state.g = nx.read_graphml('./app_data/g.graphml')

### CREATE STREAMLIT ##
def main(idx):
    initialize_session_state(idx)
    display_intro_text()
    plot_suitability_variables()
    perform_suitability_analysis()

def display_intro_text():
    st.markdown("### Phase 1: Suitability Analysis - Identify Candidate Sites for Large-scale Digester")
    st.markdown(
        "Examine the maps below, each represents a pre-selected criterion deemed crucial for determining how suitable an area is for large digesters."
        " Each area in the region is given a suitability score between 0 and 1, representing least and most suitable respectively."
        " Tip: Click the question mark icon :grey_question: on top of each map for more information."
    )

def plot_suitability_variables():
    col1, col2, col3 = st.columns(3)
    plot_variable(col1, "Farm Locations", fuzzy_farm, "The closer to feedstocks the higher the suitability.")
    plot_variable(col1, "Road Infrastructure", fuzzy_road, "The closer to roads the higher the suitability.")
    plot_variable(col1, "Water Bodies", fuzzy_water, "The further away from water bodies the higher the suitability.")
    plot_variable(col2, "Industrial Areas", fuzzy_industry, "The closer to industrial areas the higher the suitability.")
    plot_variable(col2, "Urban and Residential Areas", fuzzy_urban, "The further away from urban and residential areas the higher the suitability.")
    plot_variable(col3, "Nature and Forest", fuzzy_nature, "The further away from natural areas and water bodies the higher the suitability.")
    plot_variable(col3, "Gas Inlets", fuzzy_inlet, "The closer to inlets the higher the suitability.")
    plot_variable(col3, "Pm25", fuzzy_pm25, "The closer to pm25 the higher the suitability.")
    col3.markdown(variable_legend_html, unsafe_allow_html=True)

def plot_variable(column, title, data, help_text):
    column.markdown(f"**{title}**", help=help_text)
    column.pydeck_chart(generate_pydeck(data), use_container_width=True)

def perform_suitability_analysis():
    with st.sidebar.form("suitability_analysis_form"):
        selected_variables = st.multiselect(":one: Select Criteria", list(all_arrays.keys()))
        submit_button = st.form_submit_button("Build Suitability Map")

    if submit_button and not selected_variables:
        st.warning("No variable selected.")
        return

    if submit_button:
        hex_df = update_layer(selected_variables, all_arrays, d_to_farm)
        get_sites(hex_df, st.session_state.w, st.session_state.g, idx)
        fig = ff.create_distplot([st.session_state.all_loi['fuzzy'].tolist()], ['Distribution'], show_hist=False, bin_size=0.02)
        fig.update_layout(autosize=True, width=600, height=400)
        st.session_state.fig = fig

    st.markdown("### **Suitability Map**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Number of Candidate Sites: {len(st.session_state['all_loi'])}**")

    if st.sidebar.button(':two: Save Result & Enter Phase 2', help="Click to save the current filtered locations for further exploration in ***Phase 2: Policy Explorer***."):
        st.session_state.loi = st.session_state.all_loi
        st.switch_page("pages/2_Phase_2_Policy_Explorer.py")

    hex_df = update_layer(selected_variables, all_arrays, d_to_farm)
    layers = get_layers(hex_df)
    plot_result(st.session_state.fig)

    loi_plot = pdk.Layer(
        "H3HexagonLayer",
        st.session_state.all_loi,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        opacity=0.6,
        get_hexagon="hex9",
        get_fill_color=[0, 0, 0, 0], 
        get_line_color=[0, 255, 0],
        line_width_min_pixels=2)
    layers.append(loi_plot)
    
    deck = pdk.Deck(layers=layers, initial_view_state=VIEW_STATE, tooltip={"text": "Suitability: {fuzzy}"})
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown(variable_legend_html, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    idx = load_gdf('./app_data/h3_polygons.shp')
    main(idx)
