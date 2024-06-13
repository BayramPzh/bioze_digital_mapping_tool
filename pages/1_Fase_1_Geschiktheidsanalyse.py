# Importing standard libraries
from io import BytesIO

# Importing third-party libraries
import base64
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import pydeck as pdk
import streamlit as st
from pysal.explore import esda
from pysal.lib import weights

# Importing local application/library specific imports
from utils.cflp_function import *

#####

# Constants
PADDING = 0
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8, bearing=0, pitch=0)
DATA_PATHS = {
    'farm': './hex/h3_farm_mock_data.csv',
    'road': './hex/h3_indices_2.csv',
    'industry': './hex/h3_indices_3.csv',
    'nature': './hex/h3_indices_4.csv',
    'water': './hex/h3_indices_5.csv',
    'urban': './hex/h3_indices_6.csv',
    'inlet': './hex/h3_indices_7.csv',
}

# Generating colormap
color_mapping = generate_color_mapping(COLORMAP)

# Setting page configuration
st.set_page_config(page_title="Geschiktheids Analyse", layout="wide")

# Setting markdown
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

#####

def load_data(csv_path):
    """Function to load data from a CSV file."""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {csv_path} not found.")

def load_gdf(gdf_path):
    """Function to load a GeoDataFrame from a file."""
    try:
        return gpd.read_file(gdf_path).set_index('hex9')
    except FileNotFoundError:
        raise FileNotFoundError(f"File {gdf_path} not found.")

# Loading dataframes
d_to_farm = load_data(DATA_PATHS['farm'])
d_to_road = load_data(DATA_PATHS['road'])
d_to_industry = load_data(DATA_PATHS['industry'])
d_to_nature = load_data(DATA_PATHS['nature'])
d_to_water = load_data(DATA_PATHS['water'])
d_to_urban = load_data(DATA_PATHS['urban'])
d_to_inlet = load_data(DATA_PATHS['inlet'])

# Checking if data is loaded correctly
if d_to_farm is None or d_to_road is None or d_to_industry is None or d_to_nature is None or d_to_water is None or d_to_urban is None or d_to_inlet is None:
    print("Error loading data.")
    exit()

#####

# Fuzzify input variables
@st.cache_data
def fuzzify(df, type="close", colormap_name=color_mapping):
    df_array = np.array(df['value'])
    fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / (df_array.max() - df_array.min())) if type == "close" else np.maximum(0, (df_array - df_array.min()) / (df_array.max() - df_array.min()))
    df['fuzzy'] = fuzzified_array.round(3)
    apply_color_mapping(df, 'fuzzy', color_mapping)
    return df

# Fuzzifying dataframes
fuzzy_farm = fuzzify(d_to_farm, type='close')
fuzzy_road = fuzzify(d_to_road, type='close')
fuzzy_industry = fuzzify(d_to_industry, type='close')
fuzzy_nature = fuzzify(d_to_nature, type='far')
fuzzy_water = fuzzify(d_to_water, type='far')
fuzzy_urban = fuzzify(d_to_urban, type='far')
fuzzy_inlet = fuzzify(d_to_inlet, type='close')
# fuzzy_pm25 = fuzzify(d_to_pm25, type='close')  # or type='far',

# All arrays
all_arrays = {'Boerderijen': np.array(fuzzy_farm['fuzzy']), 
              'Weginfrastructuur': np.array(fuzzy_road['fuzzy']),
              'Stedelijke en woongebieden': np.array(fuzzy_urban['fuzzy']), 
              'Industriële gebieden': np.array(fuzzy_industry['fuzzy']), 
              'Natuur': np.array(fuzzy_nature['fuzzy']),
              'Waterlichamen': np.array(fuzzy_water['fuzzy']),
              'Gasinlaten': np.array(fuzzy_inlet['fuzzy']),
            #   'Pm25': np.array(fuzzy_pm25['fuzzy'])
            }

#####

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
        # fuzzy_df = fuzzy_df.set_index('hex9').reindex(idx.index)
        fuzzy_df = fuzzy_df.drop_duplicates(subset='hex9').set_index('hex9').reindex(idx.index)
        # st.write(fuzzy_df)
        lisa = esda.Moran_Local(fuzzy_df['fuzzy'], w, seed=42)
        # HH = fuzzy_df[(lisa.q == 1) & (lisa.p_sim < 0.01)].index.to_list()
        HH = fuzzy_df[(lisa.p_sim < 0.05)].index.to_list()
        H = g.subgraph(HH)
        subH = list(nx.connected_components(H))
        # filter_subH = [component for component in subH if len(component) > 10]
        filter_subH = [component for component in subH if len(component) > 5]
        site_idx = []
        for component in filter_subH:
            subgraph = H.subgraph(component)
            eigenvector_centrality = nx.eigenvector_centrality(subgraph, max_iter=1500)
            max_node_index = max(eigenvector_centrality, key=eigenvector_centrality.get)
            site_idx.append(max_node_index)
        st.session_state.all_loi = fuzzy_df.loc[site_idx].reset_index()
        st.write(st.session_state.all_loi)
    else:
        return None

#####

# Generate pydeck
@st.cache_resource
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
                    tooltip={"text": "Geschiktheid:" f"{{fuzzy}}"}
    )

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

variable_legend_html = generate_colormap_legend(label_left='Minst Geschikt (0)', label_right='Meest Geschikt (1)',)

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

#####

### CREATE STREAMLIT ##
def main(idx):
    initialize_session_state(idx)
    display_intro_text()
    plot_suitability_variables()
    perform_suitability_analysis()


# Initialize session state | STAP 1
def initialize_session_state(idx):
    if 'all_loi' not in st.session_state:
        st.session_state.all_loi = pd.DataFrame()
    if 'loi' not in st.session_state:
        st.session_state.loi = pd.DataFrame()
    if 'fig' not in st.session_state:
        st.session_state.fig = None
    if 'w' not in st.session_state:
        st.session_state.w = weights.Queen.from_dataframe(idx, use_index=True)
        # st.write(st.session_state.w)
    if 'g' not in st.session_state:
        st.session_state.g = nx.read_graphml('./app_data/g.graphml')
        # st.write(st.session_state.g)


### STAP 2
def display_intro_text():
    st.markdown("### Fase 1: Geschiktheidsanalyse - Potentiële Locaties voor Grootschalige Vergisters")
    st.markdown(
        "Bekijk de onderstaande kaarten, elk vertegenwoordigt een vooraf geselecteerd criterium dat essentieel wordt geacht voor het bepalen van de geschiktheid van een gebied voor grootschalige vergisters.  "
        " Elk gebied in de regio krijgt een geschiktheidsscore tussen 0 en 1, waarbij 0 het minst geschikt en 1 het meest geschikt vertegenwoordigt.  "
        "<br>Tip: Klik op het vraagtekenpictogram :grey_question: boven elke kaart voor meer informatie.",
        unsafe_allow_html=True
    )


### STAP 3
def plot_suitability_variables():
    col1, col2, col3 = st.columns(3)
    plot_variable(col1, "Boerderij locaties", fuzzy_farm, "Hoe dichter bij voedingsstoffen, hoe geschikter.")
    plot_variable(col1, "Weginfrastructuur", fuzzy_road, "Hoe dichter bij wegen, hoe geschikter.")
    plot_variable(col1, "Waterlichamen", fuzzy_water, "Hoe verder weg van waterlichamen, hoe geschikter.")
    plot_variable(col2, "Industriële Gebieden", fuzzy_industry, "Hoe dichter bij industriële gebieden, hoe geschikter.")
    plot_variable(col2, "Stedelijke en Woongebieden", fuzzy_urban, "Hoe verder weg van stedelijke en woongebieden, hoe geschikter.")
    plot_variable(col3, "Natuur en Bos", fuzzy_nature, "Hoe verder weg van natuurgebieden en waterlichamen, hoe geschikter.")
    plot_variable(col3, "Gasinlaten", fuzzy_inlet, "Hoe dichter bij inlaten, hoe geschikter.")
    # plot_variable(col3, "Pm25", fuzzy_pm25, "The closer to pm25 the higher the suitability.")
    col3.markdown(variable_legend_html, unsafe_allow_html=True)

def plot_variable(column, title, data, help_text):
    # st.write(data)
    column.markdown(f"**{title}**", help=help_text)
    column.pydeck_chart(generate_pydeck(data), use_container_width=True)


### STAP 4
def perform_suitability_analysis():
    with st.sidebar.form("suitability_analysis_form"):
        selected_variables = st.multiselect(":one: Selecteer Criteria", list(all_arrays.keys()))
        submit_button = st.form_submit_button("Bouw Geschiktheidskaart")

    if submit_button and not selected_variables:
        st.warning("Geen variabele geselecteerd.")
        return

    if submit_button:
        hex_df = update_layer(selected_variables, all_arrays, d_to_farm)
        get_sites(hex_df, st.session_state.w, st.session_state.g, idx)
        if not st.session_state.all_loi['fuzzy'].empty:
            fig = ff.create_distplot([st.session_state.all_loi['fuzzy'].tolist()], ['Distribution'], show_hist=False, bin_size=0.02)
            fig.update_layout(autosize=True, width=600, height=400)
            st.session_state.fig = fig
        else:
            st.write("st.session_state.all_loi['fuzzy'] is empty.")

    st.markdown("### **Geschiktheidskaart**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Aantal Potentiële Locaties: {len(st.session_state['all_loi'])}**")

    if st.sidebar.button(':two: Resultaat Opslaan & Ga naar Fase 2', help="Klik om de huidige gefilterde locaties op te slaan voor verder onderzoek in ***Fase 2: Beleid Verkenner***."):
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
    idx = load_gdf('./app_data/h3_pzh_polygons.shp')
    main(idx)