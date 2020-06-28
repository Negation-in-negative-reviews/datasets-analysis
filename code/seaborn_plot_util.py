import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
# LINEWIDTH = 3
# sns.set(font_scale=2.2, rc={
#         "lines.linewidth": LINEWIDTH,
#         "lines.markersize": 20,
#         "ps.useafm": True,
#         "font.sans-serif": ["Helvetica"],
#         "pdf.use14corefonts": True,
#         "text.usetex": True,
#     })
FIG_DPI = 100
sns.set(font_scale=3, style="white", rc={
    "lines.linewidth": 3,
    "lines.markersize":20,
    "ps.useafm": True,
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts" : True,
    "text.usetex": False,
    })
# sns.set(style="whitegrid")
# LINEWIDTH = 3
# MARKERSIZE = 10
# TICKLABELSIZE = 14
# LEGENDLABELSIZE = 14
# LABELSIZE = 23
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 1.0

# sns.set(font_scale=2.5, style="whitegrid", rc={
#     'figure.figsize':(8,4),
#     "lines.linewidth": 3,
#     "lines.markersize":20,
#     "ps.useafm": True,
#     "font.sans-serif": ["Helvetica"],
#     "pdf.use14corefonts" : True,
#     "text.usetex": True,
#     })

# sns.set_style("white")
filetype = ".png"
# mpl.rcParams['hatch.linewidth'] = 2.0
mpl.rcParams['hatch.color'] = "black"
err_elinewidth = 3

# def draw_histogram(data, x_label, y_label, bargraph_savepath):  
#     plt.figure()
#     plt.figure(figsize=(16,12))
#     # Affective Processes categories
   
#     df = pd.DataFrame({
#         "categories": [d[x_label] for d in data],
#         "Number of tokens": [d[y_label] for d in data],
#         "sem vals": [d["sem_value"] for d in data]        
#     })
#     print(df)
#     ax = sns.barplot(x="categories", y="Number of tokens", data=df, yerr=df["sem vals"])
#     ax.set_xlabel('')
#     ax.set_ylabel('Average no of tokens')
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#     plt.savefig(bargraph_savepath+filetype, dpi=400)


def draw_grouped_barplot(data, x_label: str, y_label: str, seaborn_hue: str, bargraph_savepath: str, 
    title: str="", y_axis_name:str="#tokens", ylim_top: float=None, amazon_data_flag=False,
                        bbox_to_anchor=None, position=None, figsize=(11, 4)):      
    print(bargraph_savepath)    
    # plt.figure()
    fig = plt.figure(figsize=figsize)
    # print(sns.get)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(111)
    data_df = pd.DataFrame({
        x_label: [d[x_label] for d in data],
        y_label: [d[y_label] for d in data],
        "sem_value": [d["sem_value"] for d in data],
        seaborn_hue: [d[seaborn_hue] for d in data]
    })
    
    data_df = data_df.sort_values(by=[x_label, seaborn_hue], ascending=[True, False])    
#     print("data_df", data_df)
    if not amazon_data_flag:
        # data_df = data_df.set_index(x_label)
        new_data_df = data_df.loc[data_df[x_label] == 'Yelp']
        for val in ['IMDB', 'SST', 'Tripadvisor']:
            new_data_df = new_data_df.append(data_df.loc[data_df[x_label] == val])
        # data_df = data_df.iloc[pd.Index(data_df[x_label]).get_indexer()]
        data_df = new_data_df
    else:
        new_data_df = data_df.replace("Cellphones", "Cell")
        new_data_df = new_data_df.replace("Luxury Beauty", "Beauty")
        new_data_df = new_data_df.replace("Automotive", "Auto")
        new_data_df = new_data_df.replace("Pet Supplies", "Pet")
        data_df = new_data_df


    print(data_df)
    colors = [(84/255, 141/255, 255/255),  (84/255, 141/255, 255/255)]*2
    subx = data_df[seaborn_hue].unique()
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df[seaborn_hue] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                yerr=dfg["sem_value"].values, 
                error_kw={"elinewidth": err_elinewidth},
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()

    if ylim_top != None:
        ax.set_ylim(0, ylim_top)
    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name
    # , labelpad=7
    )

    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')
            thisbar.set_edgecolor('black')

    pos_rev_patch = mpl.patches.Patch(facecolor=colors[0], alpha=0.9, edgecolor='black', hatch='', label='Positive review')
    neg_rev_patch = mpl.patches.Patch(facecolor=colors[1], alpha=0.9, edgecolor='black', hatch='/', label='Negative review')
    lg=plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left', bbox_to_anchor=bbox_to_anchor, frameon=False)    

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')
    
    # plt.tight_layout()
    plt.savefig(bargraph_savepath+filetype,
    # bbox_inches = "tight",
    bbox_extra_artists=(lg,), 
    dpi=FIG_DPI,
    )

def draw_grouped_barplot_four_subbars(data, x_label: str, y_label: str, 
    seaborn_hue: str, bargraph_savepath: str, title: str="", ylim_top:float = None, 
    negation: bool = False, y_axis_name="#tokens", amazon_data_flag=False,
    bbox_to_anchor=None, position=None, figsize=(11, 4)):          
    print(bargraph_savepath)
#     plt.figure()
#     plt.figure(figsize=(11,4))

    print(bargraph_savepath)    
    # plt.figure()
    fig = plt.figure(figsize=figsize)
    # print(sns.get)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(111)

    data_df = pd.DataFrame({
        x_label: [d[x_label] for d in data],
        y_label: [d[y_label] for d in data],
        "sem_value": [d["sem_value"] for d in data],
        "text_sentiment": [d["text sentiment"] for d in data],
        "review_category": [d["review category"] for d in data],
        "hue_attribute": [d["review category"]+" reviews - "+d["text sentiment"] for d in data],
    })
    data_df = data_df.sort_values(by=[x_label, "review_category", "text_sentiment"], ascending=[True, False, False])    
#     print(data_df)
    if not amazon_data_flag:
        # data_df = data_df.set_index(x_label)
        new_data_df = data_df.loc[data_df[x_label] == 'Yelp']
        for val in ['IMDB', 'SST', 'Tripadvisor']:
            new_data_df = new_data_df.append(data_df.loc[data_df[x_label] == val])
        # data_df = data_df.iloc[pd.Index(data_df[x_label]).get_indexer()]
        data_df = new_data_df
    else:
        new_data_df = data_df.replace("Cellphones", "Cell")
        new_data_df = new_data_df.replace("Luxury Beauty", "Beauty")
        new_data_df = new_data_df.replace("Automotive", "Auto")
        new_data_df = new_data_df.replace("Pet Supplies", "Pet")
        data_df = new_data_df
        
#     print("Positive sentiment")
#     print(data_df[data_df["text_sentiment"]=="positive"])
#     print("Negative sentiment")
#     print(data_df[data_df["text_sentiment"]=="negative"])
    
    colors = [(114/255, 200/255, 117/255),(209/255, 68/255, 68/255)]*2

    subx = data_df["hue_attribute"].unique()
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df["hue_attribute"] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                yerr=dfg["sem_value"].values, 
                error_kw={"elinewidth": err_elinewidth},                
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()
    if ylim_top != None:
        ax.set_ylim(0, ylim_top)

    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name)
    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')      
            thisbar.set_edgecolor('black')
        
    if negation:
        pos_sent_patch = mpl.patches.Patch(color=colors[0], label='Negation before \npositive lexicon')
        neg_sent_patch = mpl.patches.Patch(color=colors[1], label='Negation before \nnegative lexicon')
    else:
        pos_sent_patch = mpl.patches.Patch(color=colors[0], label='Positive lexicon')
        neg_sent_patch = mpl.patches.Patch(color=colors[1], label='Negative lexicon')

    pos_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='', label='Positive review')
    neg_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='/', label='Negative review')

    sentiment_bbox_to_anchor = list(bbox_to_anchor)
    sentiment_bbox_to_anchor[-1] -= 0.4
#     sentiment_bbox_to_anchor[0] -= 0.133
    sentiment_legend = plt.legend(handles=[pos_sent_patch, neg_sent_patch], bbox_to_anchor=sentiment_bbox_to_anchor, frameon=False)
#     plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left')
    plt.gca().add_artist(sentiment_legend)
    lg=plt.legend(handles=[pos_rev_patch, neg_rev_patch], bbox_to_anchor=bbox_to_anchor, frameon=False)    

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')

    # plt.tight_layout()
    plt.savefig(bargraph_savepath+filetype, 
#     bbox_inches = "tight",
    # bbox_extra_artists=(lg,), 
    dpi=FIG_DPI
    )

def draw_grouped_barplot_four_subbars_liwc(data, colors, x_label: str, y_label: str, 
    seaborn_hue: str, bargraph_savepath: str, title: str="", ylim_top:float = None, 
    liwc_cats=None, y_axis_name="#tokens", amazon_data_flag=False,
    bbox_to_anchor=None, position=None, figsize=(11, 4)):          
    print(bargraph_savepath)
    # plt.figure()
    # plt.figure(figsize=(11,4))
    fig = plt.figure(figsize=figsize)
    # print(sns.get)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(111)

    data_df = pd.DataFrame({
        x_label: [d[x_label] for d in data],
        y_label: [d[y_label] for d in data],
        "sem_value": [d["sem_value"] for d in data],
        "liwc_category": [d["liwc_category"] for d in data],
        "review category": [d["review category"] for d in data],
        "hue_attribute": [d["review category"]+" reviews - "+d["liwc_category"] for d in data],
    })
    data_df = data_df.sort_values(by=[x_label, "review category", "liwc_category"], ascending=[True, False, False])    
    print(data_df)
    if not amazon_data_flag:
        # data_df = data_df.set_index(x_label)
        new_data_df = data_df.loc[data_df[x_label] == 'Yelp']
        for val in ['IMDB', 'SST', 'Tripadvisor']:
            new_data_df = new_data_df.append(data_df.loc[data_df[x_label] == val])
        # data_df = data_df.iloc[pd.Index(data_df[x_label]).get_indexer()]
        data_df = new_data_df
    else:
        new_data_df = data_df.replace("Cellphones", "Cell")
        new_data_df = new_data_df.replace("Luxury Beauty", "Beauty")
        new_data_df = new_data_df.replace("Automotive", "Auto")
        new_data_df = new_data_df.replace("Pet Supplies", "Pet")
        data_df = new_data_df

    
    subx = data_df["hue_attribute"].unique()
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df["hue_attribute"] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                yerr=dfg["sem_value"].values, 
                error_kw={"elinewidth": err_elinewidth},
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name)
    if ylim_top != None:
        plt.ylim(0, ylim_top)

    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')
            thisbar.set_edgecolor('black')

    pos_sent_patch = mpl.patches.Patch(color=colors[0], label=liwc_cats[0])
    neg_sent_patch = mpl.patches.Patch(color=colors[1], label=liwc_cats[1])

    pos_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='', label='Positive review')
    neg_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='/', label='Negative review')

    sentiment_bbox_to_anchor = list(bbox_to_anchor)
    sentiment_bbox_to_anchor[-1] -= 0.4
    sentiment_legend = plt.legend(handles=[pos_sent_patch, neg_sent_patch], bbox_to_anchor=sentiment_bbox_to_anchor, frameon=False)
#     plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left')
    plt.gca().add_artist(sentiment_legend)
    lg=plt.legend(handles=[pos_rev_patch, neg_rev_patch], bbox_to_anchor=bbox_to_anchor, frameon=False)    

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')

    # plt.tight_layout()
    plt.savefig(bargraph_savepath+filetype, 
#     bbox_inches = "tight",
    # bbox_extra_artists=(lg,), 
    dpi=FIG_DPI
    )

    # sentiment_legend = plt.legend(handles=[pos_sent_patch, neg_sent_patch], loc='upper right')
    # plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left')
    # plt.gca().add_artist(sentiment_legend)

    # ax.tick_params(axis='x', which='major')
    # ax.tick_params(axis='y', which='major')
    # # plt.tight_layout()
    # plt.savefig(bargraph_savepath+filetype, 
    # bbox_inches = "tight",
    # dpi=FIG_DPI
    # )
