import dnaplotlib as dpl
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Ellipse, Wedge, Circle, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke
import matplotlib.patches as patches


import numpy as np

color_list = np.array([ np.array([251,166,73], dtype='int'), #Used for lactate/lldR
                        np.array([69,196,175], dtype='int'), #Used for dCas9
                        np.array([236,108,95], dtype='int'), #
                        np.array([76,139,48], dtype='int'), #Used for sgRNA, GFP, sgRNA2
                        np.array([100,25,120], dtype='int'), #Used for sRNA
                        np.array([221,204,119], dtype='int'), #Used for asRNA
                        np.array([204,121,167], dtype='int'), #
                        np.array([254, 226, 195], dtype='int'), #
                        np.array([136,34,85], dtype='int'), #
                        np.array([194, 236, 229], dtype='int'), #
                        np.array([0,0,0], dtype=int)])/255 #Used for line drawing

# Text size parameters
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    newcolor = 1 - amount * (1 - color)

    return newcolor

# Define drawing parameters
lw = 1.0 # Global line width
y_upper_lim = 30
y_lower_lim = 28

figsize_width = 10
figsize_height = 5.625
# Function for drawing small molecule glyphs
#Draw small molecule
def sbol_smallmolecule(ax, type, num, start, end, prev_end, scale, linewidth, y_pos, opts):
    """
    Drawing sbol small molecule
    """
        # Default options
    zorder_add = 0.0
    color = (1,1,1)
    edgecolor = (0,0,0)
    start_pad = 2.0
    end_pad = 20.0
    #Radius
    x_extent = 6.0
    y_extent = 6.0
    linestyle = '-'
    # Reset defaults if provided
    if opts != None:
        if 'zorder_add' in list(opts.keys()):
            zorder_add = opts['zorder_add']
        if 'edgecolor' in list(opts.keys()):
            edgecolor = opts['edgecolor']
        if 'color' in list(opts.keys()):
            color = opts['color']
        if 'start_pad' in list(opts.keys()):
            start_pad = opts['start_pad']
        if 'end_pad' in list(opts.keys()):
            end_pad = opts['end_pad']
        if 'x_extent' in list(opts.keys()):
            x_extent = opts['x_extent']
        if 'y_extent' in list(opts.keys()):
            y_extent = opts['y_extent']
        if 'linestyle' in list(opts.keys()):
            linestyle = opts['linestyle']
        if 'linewidth' in list(opts.keys()):
            linewidth = opts['linewidth']
        if 'scale' in list(opts.keys()):
            scale = opts['scale']
    # Check direction add start padding
    final_end = end
    final_start = prev_end
    
    start = prev_end+start_pad
    end = start+x_extent
    final_end = end+end_pad
    rbs_center = (start+((end-start)/2.0),y_pos)
     
    center_x = start+(end-start)/2.0
    radius = x_extent/2

    delta = radius - 0.5 * radius * np.sqrt(2)
    
    circle = Circle(rbs_center, x_extent/2.0, linewidth=linewidth, edgecolor=edgecolor, 
                facecolor=color, zorder=12+zorder_add)

    ax.add_patch(circle)
    dpl.write_label(ax, opts['label'], final_start+((final_end-final_start)/2.0), opts=opts)
    
    return start,end

def draw_lactate(ax, linelength, opts):
                                                #ax, type, num, start, end, prev_end, scale, linewidth, y_pos
    start_sbol, arrow_start = sbol_smallmolecule(ax, '', 0,   0,     0,  20,        1,     1,         20   ,opts)

    padding = 2
    length = 15
    arrow1 = Line2D([arrow_start+padding,arrow_start+linelength], [20,20], linewidth=lw, 
                color=opts['color'], zorder=12)
    ax.add_line(arrow1)

# Define and draw original asRNA circuit
# Promoter
p_Const = {'type':'Promoter', 'name':'const.', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'label':'const.', 'label_y_offset':-6, 'label_x_offset':-6}}
p_ALPaGA = {'type':'Promoter', 'name':'ALPaGA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'label':'ALPaGA', 'label_y_offset':-6, 'label_x_offset':-6}}

# RBS
rbs = {'type':'RBS', 'name':'rbs', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'start_pad':6}}
# CDS
g_lldR = {'type':'CDS', 'name':'lldR', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[0], 'label':'lldR', 'label_y_offset':0, 'label_x_offset':-2, 'x_extent':24}}
g_dCas9 = {'type':'CDS', 'name':'dCas', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[1], 'label':'dCas', 'label_y_offset':0, 'label_x_offset':-2,'x_extent':24}}

g_GFP_off = {'type':'CDS', 'name':'GFP', 'fwd':True, 'opts':{'linewidth':lw, 'color':np.array([220,220,220])/255, 'label':'GFP', 'label_y_offset':0, 'label_x_offset':-2,'x_extent':24}}
g_GFP_on = {'type':'CDS', 'name':'GFP', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[3], 'label':'GFP', 'label_y_offset':0, 'label_x_offset':-2,'x_extent':24}}

g_sgRNA = {'type':'CDS', 'name':'sgRNA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[3], 'label':'sgRNA', 'label_y_offset':0,'label_x_offset':-2, 'x_extent':24}}
g_asRNA = {'type':'CDS', 'name':'asRNA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[5], 'label':'asRNA', 'label_y_offset':0,'label_x_offset':-2, 'x_extent':24}}

# Terminator
t_Ter = {'type':'Terminator', 'name':'Ter', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'label':'', 'label_y_offset':-8, 'start_pad':-1}}

# Define regulatory interactions
# arc_height minimum is 18, or else it looks weird
#Cascomplex1
sgrna_repression = {'type': 'Repression','from_part':g_sgRNA, 'to_part':g_GFP_off, 'opts':{'color':color_list[1], 'linewidth':lw, 'arc_height':20}}
dcas_repression = {'type': 'Repression', 'from_part': g_dCas9, 'to_part': g_GFP_off, 'opts':{'color':color_list[1], 'linewidth':lw, 'arc_height':20}}

#Anti-sense repression
asrna_repression = {'type': 'Repression', 'from_part': g_asRNA, 'to_part': g_sgRNA, 'opts':{'color':color_list[5], 'linewidth':lw,
 'arc_height':-25, 'arc_height_start':-12, 'arc_height_end':-12, 'arc_end_x_offset':0}}

#Lactate promoter
lldr_repression = {'type': 'Repression', 'from_part': g_lldR, 'to_part': p_ALPaGA, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25}}
lldr_activation = {'type': 'Activation', 'from_part': g_lldR, 'to_part': p_ALPaGA, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25}}

# Define and draw the DNA assembly
# 'Normal' asRNA circuit
reg1_off = [sgrna_repression, dcas_repression, lldr_repression]
reg1_on = [lldr_activation, asrna_repression]

design1_off = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_Const, rbs, g_GFP_off, t_Ter]
design1_on = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_Const, rbs, g_GFP_on, t_Ter]
dr = dpl.DNARenderer()

# Draw the OFF state
fig= plt.figure(figsize=(figsize_width, figsize_height), tight_layout=True)
ax = fig.add_subplot(111)

start, end = dr.renderDNA(ax, design1_off, dr.SBOL_part_renderers(), 
	                      regs=reg1_off, reg_renderers=dr.std_reg_renderers())
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/asRNA_circuit_off.svg', bbox_inches='tight')
plt.close()

# Draw the ON state
fig= plt.figure(figsize=(figsize_width, figsize_height), tight_layout=True)
ax = fig.add_subplot(111)

start, end = dr.renderDNA(ax, design1_on, dr.SBOL_part_renderers(),
                            regs=reg1_on, reg_renderers=dr.std_reg_renderers())

lactate_opts = {'label':"lactate", 'label_y_offset':12,'label_x_offset':-5, 'edgecolor':color_list[0]*0.8, 'color':color_list[0], 'linewidth':lw}
draw_lactate(ax, 17, lactate_opts)
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/asRNA_circuit_on.svg', bbox_inches='tight')
plt.close()

# FFN asRNA circuit
# Define more parts/interactions
p_ALPaGA2 = {'type':'Promoter', 'name':'ALPaGA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'label':'ALPaGA', 'label_y_offset':-6, 'label_x_offset':-6}}

lldr_repression2 = {'type': 'Repression', 'from_part': g_lldR, 'to_part': p_ALPaGA2, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25}}
lldr_activation2 = {'type': 'Activation', 'from_part': g_lldR, 'to_part': p_ALPaGA2, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25}}

reg2_off = [sgrna_repression, dcas_repression, lldr_repression, lldr_repression2]
reg2_on = [lldr_activation, lldr_activation2, asrna_repression]

design2_off = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_off, t_Ter]
design2_on = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_on, t_Ter]
# Draw the FFN circuit - OFF state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design2_off, dr.SBOL_part_renderers(),
                            regs=reg2_off, reg_renderers=dr.std_reg_renderers())
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/FFN_asRNA_circuit_off.svg', bbox_inches='tight')
plt.close()

# Draw the FFN circuit - ON state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design2_on, dr.SBOL_part_renderers(),
                            regs=reg2_on, reg_renderers=dr.std_reg_renderers())

lactate_opts = {'label':"lactate", 'label_y_offset':12,'label_x_offset':-5, 'edgecolor':color_list[0]*0.8, 'color':color_list[0], 'linewidth':lw}
draw_lactate(ax, 17, lactate_opts)
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/FFN_asRNA_circuit_on.svg', bbox_inches='tight')
plt.close()


# sRNA leak dampening circuit
# Define more parts/interactions
g_sRNA = {'type':'CDS', 'name':'sRNA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[2], 'label':'sRNA', 'label_y_offset':0,'label_x_offset':-2, 'x_extent':24}}
g_GFP_partial = {'type':'CDS', 'name':'GFP', 'fwd':True, 'opts':{'linewidth':lw, 'color':lighten_color(color_list[3],0.7), 'label':'GFP', 'label_y_offset':0,'label_x_offset':-2, 'x_extent':24}}

srna_repression = {'type': 'Repression', 'from_part': g_sRNA, 'to_part': g_GFP_off, 'opts':{'color':color_list[2], 'linewidth':lw, 'arc_height':-25, 'arc_height_start':-12, 'arc_height_end':-12, 'arc_end_x_offset':0}}
srna_repression2 = {'type': 'Repression', 'from_part': g_sRNA, 'to_part': g_GFP_partial, 'opts':{'color':color_list[2], 'linewidth':lw, 'arc_height':-25, 'arc_height_start':-12, 'arc_height_end':-12, 'arc_end_x_offset':0}}

reg3_off = [sgrna_repression, dcas_repression, lldr_repression, lldr_repression2, srna_repression]
reg3_on = [lldr_activation, lldr_activation2, asrna_repression, srna_repression]

design3_off = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_off, t_Ter, p_Const, g_sRNA, t_Ter]
design3_on = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_partial, t_Ter, p_Const, g_sRNA, t_Ter]

# Draw the sRNA circuit - OFF state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design3_off, dr.SBOL_part_renderers(),
                            regs=reg3_off, reg_renderers=dr.std_reg_renderers())
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/sRNA_asRNA_circuit_off.svg', bbox_inches='tight')
plt.close()

# Draw the sRNA circuit - ON state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design3_on, dr.SBOL_part_renderers(),
                            regs=reg3_on, reg_renderers=dr.std_reg_renderers())

lactate_opts = {'label':"lactate", 'label_y_offset':12,'label_x_offset':-5, 'edgecolor':color_list[0]*0.8, 'color':color_list[0], 'linewidth':lw}
draw_lactate(ax, 17, lactate_opts)
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/sRNA_asRNA_circuit_on.svg', bbox_inches='tight')

# sRNA leak dampening switch circuit
# Define more parts/interactions
p_ALPaGA3 = {'type':'Promoter', 'name':'ALPaGA', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[-1], 'label':'ALPaGA', 'label_y_offset':-6, 'label_x_offset':-6}}
g_sgRNA2 = {'type':'CDS', 'name':'sgRNA2', 'fwd':True, 'opts':{'linewidth':lw, 'color':color_list[6], 'label':'sgRNA2', 'label_y_offset':0,'label_x_offset':-1, 'x_extent':24}}

sgrna_repression2 = {'type': 'Repression', 'from_part': g_sgRNA2, 'to_part': g_sRNA, 'opts':{'color':color_list[6], 'linewidth':lw, 'arc_height':20, 'arc_height_start':12, 'arc_height_end':12, 'arc_end_x_offset':0}}
dcas_repression2 = {'type': 'Repression', 'from_part': g_dCas9, 'to_part': p_ALPaGA3, 'opts':{'color':color_list[1], 'linewidth':lw, 'arc_height':25, 'arc_height_start':12, 'arc_height_end':12, 'arc_end_x_offset':0}}
lldr_repression3 = {'type': 'Repression', 'from_part': g_lldR, 'to_part': p_ALPaGA3, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25, 'arc_end_x_offset':0}}
lldr_activation3 = {'type': 'Activation', 'from_part': g_lldR, 'to_part': p_ALPaGA3, 'opts':{'color':color_list[0], 'linewidth':lw, 'arc_height':25, 'arc_end_x_offset':0}}

reg4_off = [sgrna_repression, dcas_repression, lldr_repression, lldr_repression2, lldr_repression3, srna_repression]
reg4_on = [lldr_activation, lldr_activation2, lldr_activation3, asrna_repression, sgrna_repression2]

design4_off = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_off, t_Ter, p_Const, g_sRNA, t_Ter, p_ALPaGA3, g_sgRNA2, t_Ter]
design4_on = [p_Const, rbs, g_lldR, t_Ter, p_Const, rbs, g_dCas9, t_Ter, p_Const, g_sgRNA, t_Ter, p_ALPaGA, g_asRNA, t_Ter, p_ALPaGA2, rbs, g_GFP_on, t_Ter, p_Const, g_sRNA, t_Ter, p_ALPaGA3, g_sgRNA2, t_Ter]

# Draw the sRNA leak dampening circuit - OFF state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design4_off, dr.SBOL_part_renderers(),
                            regs=reg4_off, reg_renderers=dr.std_reg_renderers())
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/sRNAswitch_asRNA_circuit_off.svg', bbox_inches='tight')
plt.close()

# Draw the sRNA leak dampening circuit - ON state
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

start, end = dr.renderDNA(ax, design4_on, dr.SBOL_part_renderers(),
                            regs=reg4_on, reg_renderers=dr.std_reg_renderers())

lactate_opts = {'label':"lactate", 'label_y_offset':12,'label_x_offset':-5, 'edgecolor':color_list[0]*0.8, 'color':color_list[0], 'linewidth':lw}
draw_lactate(ax, 17, lactate_opts)
ax.set_xlim([start, end])
ax.set_ylim([-y_lower_lim,y_upper_lim])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure
fig.savefig('figures/sRNAswitch_asRNA_circuit_on.svg', bbox_inches='tight')
plt.close()



