import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects.   
'''

ave_G = 6  # fixed costs per G
ave_Gm = 5  # for inclusion in graph
ave_Gd = 4
G_aves = [ave_Gm, ave_Gd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val

class Cgraph(CPP):  # graph or generic PP of any composition

    plevels = list  # plevel_t[1]s is summed from alt_graph_, sub comp support, agg comp suppression?
    fds = list  # prior forks in plevels, then player fds in plevel
    valt = lambda: [0, 0]
    nvalt = lambda: [0, 0]  # from neg open links
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    link_ = list  # all evaluated external graph links, nested in link_layers? open links replace alt_node_
    node_ = list  # graph elements, root of layers and levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # higher-order segG or graph
    alt_graph_ = list


def agg_recursion(root, G_, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    ivalt = root.valts[-1]
    mgraph_, dgraph_ = form_graph_(root, G_, fder=0)  # PP cross-comp and clustering

    # intra graph:
    if root.valt[0] > ave_sub * root.rdn:
        sub_rlayers, rvalt = sub_recursion_g(mgraph_, root.valt, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
    if root.valt[1] > ave_sub * root.rdn:
        sub_dlayers, dvalt = sub_recursion_g(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers

    # cross graph:
    root.mlevels += mgraph_; root.dlevels += dgraph_
    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        adj_val = root.valt[fd] - ivalt[fd]  # or valts[-1][fa]?
        # recursion if adjusted val:
        if (adj_val > G_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, fseg=fseg)  # cross-comp graphs


def form_graph_(root, G_, fder):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph

    for G in G_:  # initialize mgraph, dgraph as roott per G, for comp_G_
        for i in 0,1:
            graph = [[G], [], [0,0]]  # proto-GG: [node_, meds_, valt]
            G.roott[i] = graph

    comp_G_(G_, fder)  # cross-comp all graphs within rng, graphs may be segs
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][0])>1: mgraph_ += [G.roott[0]]  # root = [node_, valt] for cluster_node_layer eval, + link_nvalt?
        if len(G.roott[1][0])>1: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            eval_med_layer(graph_= graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd, fder)  # sum proto-graph node_ params in graph
            plevels = deepcopy(graph_[0].plevels); fds = graph_[0].fds  # same for all nodes?
            for graph in graph_[1:]:
                sum_plevels(plevels, graph.plevels, fds, graph.fds)  # each plevel is (caTree, valt)
            root.plevels = plevels

    return mgraph_, dgraph_


def comp_G_(G_, fder):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):

        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign
            if fder:  # integrate into main sequence?
                comp_derG(_G.plevels[-1], G.plevels[-1])  # G is derG
                continue
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = (_G.xn-_G.x0)/2 - (G.xn-G.x0)/2; dy = (_G.yn-_G.y0)/2 - (G.yn-G.y0)/2
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, max depends on combined value:
            if distance <= ave_rng * ((sum(_G.valt)+sum(G.valt)) / (2*sum(G_aves))):

                mplevel, dplevel = comp_plevels(_G.plevels, G.plevels, _G.fds, G.fds)
                valt = [mplevel[1] - ave_Gm, dplevel[1] - ave_Gd]  # valt is already normalized, *= link rdn?
                derG = Cgraph(
                    plevels=[mplevel, dplevel], x0=min(_G.x0,G.x0), xn=max(_G.xn,G.xn), y0=min(_G.y0,G.y0), yn=max(_G.yn,G.yn), valt=valt, node_=[_G,G])
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd in 0,1:
                    if valt[fd] > 0:  # alt fork is redundant, no support?
                        for node, (graph, meds_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                meds_ += [[derG.node_[0] if derG.node_[1] is node else derG.node_[1] for derG in node.link_]]  # immediate links
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]


def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, meds_, valt = graph
    save_node_, save_meds_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for G, med_node_ in zip(node_, meds_):  # G: node or sub-graph
        mmed_node_ = []  # __Gs that mediate between Gs and _Gs
        for _G in med_node_:
            for derG in _G.link_:
                if derG not in G.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derG.node_[0].link_ if derG.node_[0] is not _G else derG.node_[1].link_
                    for _derG in med_link_:
                        if G in _derG.node_ and _derG not in G.link_:  # __G mediates between _G and G
                            G.link_ += [_derG]
                            adj_val = _derG.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            G.valt[fd] += adj_val; _G.valt[fd] += adj_val  # valts not updated
                            valt[fd] += adj_val; _G.roott[fd][2][fd] += adj_val  # root is not graph yet
                            __G = _derG.node_[0] if _derG.node_[0] is not _G else _derG.node_[1]
                            if __G not in mmed_node_:  # not saved via prior _G
                                mmed_node_ += [__G]
                                adj_Val += adj_val
        if G.valt[fd]>0:
            # G remains in graph
            save_node_ += [G]; save_meds_ += [mmed_node_]  # mmed_node_ may be empty

    for G, mmed_ in zip(save_node_, save_meds_):  # eval graph merge after adjusting graph by mediating node layer
        add_mmed_= []
        for _G in mmed_:
            _graph = _G.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
                _node_, _meds_, _valt = _graph
                for _node, _meds in zip(_node_, _meds_):  # merge graphs, ignore _med_? add direct links:
                    for derG in _node.link_:
                        __G = derG.node_[0] if derG.node_[0] is not _G else derG.node_[1]
                        if __G not in add_mmed_ + mmed_:  # not saved via prior _G
                            add_mmed_ += [__G]
                            adj_Val += derG.valt[fd] - ave_agg
                valt[fd] += _valt[fd]
                graph_.remove(_graph)
        mmed_ += add_mmed_

    graph[:] = [save_node_,save_meds_,valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmed_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph


def sub_recursion_g(graph_, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    sub_valt = [0,0]
    for graph in graph_:
        if fd:
            node_ = []  # positive links within graph
            for node in graph.node_:
                for link in node.link_:
                    if link.valt[1]>0 and link not in node_:
                        node_ += [link]
        else: node_ = graph.node_

        # rng+|der+ if top player valt[fd], for plevels[:-1]|players[-1][fd=1]:  (graph.valt eval for agg+ only)
        if graph.plevels[-1][0][-1][0][-1][1][fd] > G_aves[fd] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_, fder=fd)  # cross-comp and clustering cycle
            # rng+:
            if graph.valt[0] > ave_sub * graph.rdn:  # >cost of calling sub_recursion and looping:
                sub_rlayers, valt = sub_recursion_g(sub_mgraph_, graph.valt, fd=0)
                rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
                graph.rlayers = [sub_mgraph_] + [sub_rlayers]
            # der+:
            if graph.valt[1] > ave_sub * graph.rdn:
                sub_dlayers, valt = sub_recursion_g(sub_dgraph_, graph.valt, fd=1)
                dvalt = sum(valt); graph.valt[1] += dvalt; sub_valt[1] += dvalt
                graph.dlayers = [sub_dgraph_] + [sub_dlayers]

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers, graph.dlayers]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers_t, sub_valt


def sum2graph_(G_, fd, fder):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fsub

    graph_ = []  # new graph_
    for G in G_:
        node_, meds_, valt = G
        node = node_[0]  # init graph with 1st node:
        graph = Cgraph( plevels=deepcopy(node.plevels), fds=deepcopy(node.fds), valt=deepcopy(node.valt), valts=deepcopy(node.valts),
                        x0=node.x0, xn=node.xn, y0=node.y0, yn=node.yn, node_ = node_, meds_ = meds_)

        derG = node.link_[0]  # init new_plevel with 1st derG:
        graph.valt[0] += derG.valt[fd]; graph.valts += [[deepcopy(derG.valt[fd])]]  # add new level of valt, cis only
        new_plevel = derG.plevels[fd]; derG.roott[fd] = graph
        for derG in node.link_[1:]:
            sum_plevel(new_plevel, derG.plevels[fd])  # accum derG in new plevel
            graph.valt[0] += derG.valt[fd]; graph.valts[-1][0] += derG.valt[fd]
            derG.roott[fd] = graph
        for node in node_[1:]:
            graph.x0=min(graph.x0, node.x0); graph.xn=max(graph.xn, node.xn); graph.y0=min(graph.y0, node.y0); graph.yn=max(graph.yn, node.yn)
            # accum params:
            sum_plevels(graph.plevels, node.plevels, graph.fds, node.fds)  # same for fsub
            for derG in node.link_:
                sum_plevel(new_plevel, derG.plevels[fd])  # accum derG, add to graph when complete
                valt[0] += derG.valt[fd]; graph.valts[-1][0] += derG.valt[fd]
                derG.roott[fd] = graph
                # link_ = [derG]?
            for Val, val in zip(graph.valt, node.valt): Val+=val
            for Valt, valt in zip(graph.valts, node.valts):
                Valt[0] += valt[0]; Valt[1] += valt[1]
        graph_ += [graph]
        graph.plevels += [new_plevel]
    # haven't review below yet
    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        Alt_plevels = []  # Alt_players if fsub
        for node in graph.node_:
            for derG in node.link_:
                for G in derG.node_:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph
                            # der+: plevels[-1] += player, rng+: players[-1] = player
                            # add sum alt valts
                            if fder:
                                if Alt_plevels: sum_plevel(Alt_plevels, alt_graph.plevels[-1])  # same-length caTrees?
                                else:           Alt_plevels = deepcopy(alt_graph.plevels[-1])
                            else:  # agg+
                                if Alt_plevels: sum_plevels(Alt_plevels, alt_graph.plevels, alt_graph.fds, alt_graph.fds)  # redundant fds
                                else:           Alt_plevels = deepcopy(alt_graph.plevels)
                            graph.alt_graph_ += [alt_graph]
        if graph.alt_graph_:
            graph.alt_graph_ += [Alt_plevels]  # temporary storage

    for graph in graph_:  # 3rd pass: add alt fork to each graph plevel, separate to fix nesting in 2nd pass
        if graph.alt_graph_:
            Alt_plevels = graph.alt_graph_.pop()
        else: Alt_plevels = []
    # add sum valts:
    if fder:
        for playerst, aplayerst in zip(graph.plevels[-1], Alt_plevels):  # each plevel is caTree
            if playerst and aplayerst:  # else empty
                for (cptuples, sfds, cvalt), (aptuples, afds, avalt) in zip(playerst[0], aplayerst[0]):
                    cvalt[0] += avalt[0]; avalt[1] += avalt[1]
                    cptuples += aptuples  # player Tree leaves
    else:
        for (cplayers, cvalt), (aplayers, avalt) in zip(graph.plevels, Alt_plevels):
            cvalt[0] += avalt[0]; avalt[1] += avalt[1]
            cplayers += aplayers  # plevel Tree leaves

    return graph_

def comp_plevels(_plevels, plevels, _fds, fds):
    '''
    plevels structure, singular form is a tuple, plural is a list:
    plevel = caForks, valt
    caFork = players, val, fds
    player = caforks, valt
    cafork = ptuples, val
    '''
    mplevel, dplevel = [],[]  # fd plevels, each cis+alt, same as new_caT
    mval, dval = 0,0  # m,d in new plevel, else c,a
    iVal = ave_G  # to start loop:

    for _plevel, plevel, _fd, fd in zip(reversed(_plevels[0]), reversed(plevels[0]), _fds, fds):  # caTree is a plevel
        if iVal < ave_G or _fd != fd:  # top-down, comp if higher plevels match, same agg+
            break
        mplevel, dTree = [],[]; mtval, dtval = 0, 0

        for _caForks, caForks in zip(_plevel[0], plevel[0]):  # bottom-up alt+, pass-through fds
            mforks, dforks = [],[]; mlval, dlval = 0,0

            if _players and _players:
                mplayer, dplayer = comp_players(_players[0], players[0])  # caTree2 is a player
                mplayers += [mplayer]; dplayers += [dplayer]
                mlval += mplayer[1][0]; dlval += dplayer[1][1]
            else:
                mplayers += [[]]; dplayers += [[]]  # to align same-length trees for comp and sum
            # pack fds:
            mTree += [[mplayers, players[1], mlval]]; dTree += [[dplayers, players[1], dlval]]
            mtval += mlval; dtval += dlval
        # merge Trees:
        mplevel += mTree; dplevel += mTree  # merge Trees in candidate plevels
        mval += mtval; dval += dtval
        iVal = mval+dval  # after 1st loop

    return [mplevel,mval], [dplevel,dval]  # always single new plevel


def comp_players(_playerst, playerst):  # unpack and compare layers from der+;  plevels ( caTree1 ( players ( caTree2 ( ptuples

    mplayer, dplayer = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mVal, dVal = 0,0  # m,d in new player, else c,a
    _players, _fds, _ = _playerst
    players, fds, _ = playerst

    for _caTreet, caTreet in zip(_players, players):
        mTree, dTree = [],[]; mtval, dtval = 0,0

        for _ptuplest, ptuplest in zip(_caTreet[0], caTreet[0]):  # bottom-up alt+, pass-through fds
            if _ptuplest and ptuplest:
                mtuples, dtuples, mval, dval = comp_ptuples(ptuplest[0], ptuplest[0], _fds, fds)
                mTree += [[mtuples, mval]]; dTree += [[dtuples, dval]]
                mtval += mval; mtval += dval
            else:
                mTree += [[]]; dTree += [[]]
        # merge Trees:
        mplayer += mTree; dplayer += dTree
        mVal += mtval; dVal += dtval

    return [mplayer,mVal], [dplayer,dVal]  # single new lplayer


def comp_ptuples(_ptuples, ptuples, _fds, fds):  # unpack and compare der layers, if any from der+

    mptuples, dptuples = [],[]
    mval, dval = 0,0

    for _ptuple, ptuple, _fd, fd in zip(ptuples, ptuples, _fds, fds):  # bottom-up der+, pass-through fds
        if _fd == fd:
            mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
            mptuples +=[mtuple]; mval+=mtuple.val
            dptuples +=[dtuple]; dval+=dtuple.val
        else:
            break  # comp same fds

    return mptuples, dptuples, mval, dval


def sum_plevels(pLevels, plevels, Fds, fds):

    for CaTreet, caTreet, Fd, fd in zip_longest(pLevels, plevels, Fds, fds, fillvalue=[]):  # loop top-down, selective comp depth, same agg+?
        if Fd==fd:
            if caTreet:
                if CaTreet: sum_plevel(CaTreet, caTreet)
                else:       pLevels.append(deepcopy(caTreet))
        else:
            break

# separate to sum derGs
def sum_plevel(CaTreet, caTreet):

    CaTree, tValt = CaTreet; caTree, tvalt = caTreet  # players tree

    for Playerst, playerst in zip(CaTree, caTree):
        if Playerst and playerst:

            Players, Fds, Valt = Playerst
            players, fds, valt = playerst
            if isinstance(valt, list): Valt[0] += valt[0]; Valt[1] += valt[1]  # G
            else: Valt += valt  # fd val in derG

            for Catreet, catreet in zip_longest(Players, players, fillvalue=[]):
                if catreet:
                    if Catreet: sum_player(Catreet, catreet, Fds, fds, fneg=0)
                    else:       Players += [deepcopy(catreet)]

        if isinstance(tvalt, list): tValt[0] += Valt[0]; tValt[1] += Valt[1]  # G
        else: tValt += Valt  # fd val in derG
        #? tValt += Valt; tvalt += valt

# draft
def sum_player(CaTreet, caTreet, Fds, fds, fneg=0):  # accum layers while same fds

    CaTree, Valt = CaTreet; caTree, valt = caTreet
    Valt[0] += valt[0]; Valt[1] += valt[1]

    for Ptuplest, ptuplest in zip(CaTree, caTree):
        if Ptuplest and ptuplest:  # they may empty due to we pack empty list into tree in comp_players
            Ptuples, Val = Ptuplest; ptuples, val = ptuplest  # ptuples are pack with their val
            for i, (Ptuple, ptuple, Fd, fd) in enumerate( zip_longest(Ptuples, ptuples, Fds, fds, fillvalue=[])):
                if Fd==fd:
                    if ptuple:
                        if Ptuple:
                            sum_ptuple(Ptuple, ptuple, fneg=0)
                            Val += val
                        else:      Ptuples += [deepcopy(ptuple)]
                else:
                    break


