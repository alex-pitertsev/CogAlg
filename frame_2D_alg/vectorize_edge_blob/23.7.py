def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.derH
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CQ(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_parH(root.H[0].H[fd].derH, sub_node.derH)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root
        else:
            break

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH in blob

    # add RVal=0, DVal=0 to return?
    term = 1
    for PP in PP_:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P_) > ave_nsub:
            term = 0
            sub_recursion(PP)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only
        '''        
        for i, sG_ in enumerate(sG_t):
            val,rdn = 0,0
            for sub_G in sG_:
                val += sum(sub_G.valt); rdn += sum(sub_G.rdnt)
            Val = Valt[fd]+val; Rdn = Rdnt[fd]+rdn
        or
        Val = Valt[fd] + sum([sum(G.valt) for G in sG_])
        Rdn = Rdnt[fd] + sum([sum(G.rdnt) for G in sG_])
        '''
# old:
def op_parT(_graph, graph, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    _parT, parT = _graph.parT, graph.parT

    if fcomp:
        dparT,valT,rdnT = comp_unpack(_parT, parT, rn=1)
        return dparT,valT,rdnT
    else:
        _valT, valT = _graph.valT, graph.valT
        _rdnT, rdnT = _graph.rdnT, graph.rdnT
        for i in 0,1:
            sum_unpack([_parT[i], _valT[i], _rdnT[i]], [parT[i], valT[i],rdnT[i]])

# same as comp|sum unpack?:
def op_ptuple(_ptuple, ptuple, fcomp, fd=0, fneg=0):  # may be ptuple, vertuple, or ext

    aveG = G_aves[fd]
    if fcomp:
        dtuple=CQ(n=_ptuple.n)  # + ptuple.n / 2: average n?
        rn = _ptuple.n/ptuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, d_didx, last_i, last_idx = 0,0,-1,-1

    for _i, _didx in enumerate(_ptuple.Q):  # i: index in Qd: select param set, idx: index in full param set
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(ptuple.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx == idx:
                _par = _ptuple.Qd[_i]; par = ptuple.Qd[_i+i]
                if fcomp:  # comp ptuple
                    if ptuple.Qm: val =_par+par if fd else _ptuple.Qm[_i]+ptuple.Qm[_i+i]
                    else:         val = aveG+1  # default comp for 0der pars
                    if val > aveG:
                        if isinstance(par,list):
                            if len(par)==4: m,d = comp_aangle(_par,par)
                            else: m,d = comp_angle(_par,par)
                        else: m,d = comp_par(_par, par*rn, aves[idx], finv = not i and not ptuple.Qm)
                            # finv=0 if 0der I
                        dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.Q+=[d_didx+_didx]
                        dtuple.valt[0]+=m; dtuple.valt[1]+=d  # no rdnt, rdn = m>d or d>m?)
                else:  # sum ptuple
                    D, d = _ptuple.Qd[_i], ptuple.Qd[_i+i]
                    if isinstance(d, list):  # angle or aangle
                        for j, (P,p) in enumerate(zip(D,d)): D[j] = P-p if fneg else P+p
                    else: _ptuple.Qd[i] += -d if fneg else d
                    if _ptuple.Qm:
                        mpar = ptuple.Qm[_i+i]; _ptuple.Qm[i] += -mpar if fneg else mpar
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif fcomp:
                if _idx < idx: d_didx+=didx  # no dpar per _par
            else: # insert par regardless of _idx < idx:
                _ptuple.Q.insert[idx, didx+d_didx]
                _ptuple.Q[idx+1] -= didx+d_didx  # reduce next didx
                _ptuple.Qd.insert[idx, ptuple.Qd[idx]]
                if _ptuple.Qm: _ptuple.Qm.insert[idx, ptuple.Qm[idx]]
                d_didx = 0
            if _idx < idx: break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1
        _idx += 1
    if fcomp: return dtuple


def form_PP_t(P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence_PP s
        for P in P_:
            if not P.root_tH[-1][fd]:  # else already packed in qPP
                qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers?
                P.root_tH[-1][fd] = qPP; val = 0
                uplink_ = P.link_tH[-1][fd]
                uuplink_ = []  # next layer of links
                while uplink_:
                    for derP in uplink_:
                        _P = derP._P; _qPP = _P.root_tH[-1][fd]
                        if _qPP:
                            if _qPP is not qPP:  # _P may be added to qPP via other downlinked P
                                val += _qPP[1]  # merge _qPP in qPP:
                                for qP in _qPP[0]:
                                    qP.root_tH[-1][fd] = qPP; qPP[0] += [qP]  # append qP_
                                qPP_.remove(_qPP)
                        else:
                            qPP[0] += [_P]  # pack bottom up
                            _P.root_tH[-1][fd] = qPP
                            val += derP.valt[fd]
                            uuplink_ += derP._P.link_tH[-1][fd]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP += [val, ave + 1]  # ini reval=ave+1, keep qPP same object for ref in P.roott
                qPP_ += [qPP]

        # prune qPPs by mediated links vals:
        rePP_ = reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]

        PP_t += [CPP_]  # least one PP in rePP_, which would have node_ = P_

    return PP_t  # add_alt_PPs_(graph_t)?

# draft:
def merge_PP(PP, _PP, fd, fder):

    node_=PP.node_
    for _node in _PP.node_:
        if _node not in node_:
            node_ += [_node]
            _node.root_tt[-1][fder][fd] = PP  # reassign root
    sum_derH([PP.derH, PP.valt, PP.rdnt], [_PP.derH, _PP.valt, _PP.rdnt], base_rdn=0)

    Y0,Yn,X0,Xn = PP.box; y0,yn,x0, xn = _PP.box
    PP.box = [min(X0,x0),max(Xn,xn),min(Y0,y0),max(Yn,yn)]
    # mask__, ptuple as etc.

def med_eval(last_link_, old_link_, med_valH, fd):  # recursive eval of mediated link layers, in form_graph only?

    curr_link_ = []; med_val = 0
    # compute med_valH, layer= val of links mediated by incremental number of nodes:

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += np.sum(_link.valT[fd])
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH

# currently not used:

def sum_unpack(Q,q):  # recursive unpack of two pairs of nested sequences, to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # alternating rngH( derH( rngH... nesting, down to ptuple|val|rdn
    for i, (Ele,Val,Rdn, ele,val,rdn) in enumerate(zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[])):
        if ele:
            if Ele:
                if isinstance(val,list):  # element is layer or fork
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # ptuple
                    Val_[i] += val; Rdn_[i] += rdn
                    sum_ptuple(Ele, ele)
            else:
                Que += [deepcopy(ele)]; Val_+= [deepcopy(val)]; Rdn_+= [deepcopy(rdn)]

def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]  # alternating rngH( derH( rngH.. nesting,-> ptuple|val|rdn

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:
                # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [int(mval<dval),int(mval>=dval)]  # to use np.sum

            for i in 0,1:  # adds nesting per recursion
                DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]

    return DerT,ValT,RdnT

def add_unpack(H, incr):  # recursive unpack hierarchy of unknown nesting to add input
    # new_H = []
    for i, e in enumerate(H):
        if isinstance(e,list):
            add_unpack(e,incr)
        else: H[i] += incr
    return H

def last_add(H, i):  # recursive unpack hierarchy of unknown nesting to add input
    while isinstance(H,list):
        H=H[-1]
    H+=i

def unpack(H):  # recursive unpack hierarchy of unknown nesting
    while isinstance(H,list):
        last_H = H
        H=H[-1]
    return last_H

def nest(P, ddepth=2):  # default ddepth is nest 2 times: tuple->layer->H, rngH is ptuple, derH is 1,2,4.. ptuples'layers?

    # fback adds alt fork per layer, may be empty?
    # agg+ adds depth: number brackets before the tested bracket: P.valT[0], P.valT[0][0], etc?

    if not isinstance(P.valT[0],list):
        curr_depth = 0
        while curr_depth < ddepth:
            P.derT[0]=[P.derT[0]]; P.valT[0]=[P.valT[0]]; P.rdnT[0]=[P.rdnT[0]]
            P.derT[1]=[P.derT[1]]; P.valT[1]=[P.valT[1]]; P.rdnT[1]=[P.rdnT[1]]
            curr_depth += 1

        if isinstance(P, CP):
            for derP in P.link_t[1]:
                curr_depth = 0
                while curr_depth < ddepth:
                    derP.derT[0]=[derP.derT[0]]; derP.valT[0]=[derP.valT[0]]; derP.rdnT[0]=[derP.rdnT[0]]
                    derP.derT[1]=[derP.derT[1]]; derP.valT[1]=[derP.valT[1]]; derP.rdnT[1]=[derP.rdnT[1]]
                    curr_depth += 1

def form_graph_(G_, fder):  # form list graphs and their aggHs, G is node in GG graph

    mnode_, dnode_ = [],[]  # Gs with >0 +ve fork links:

    for G in G_:
        if G.link_tH[0]: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_tH[1]: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = init_graph(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_ += [_G]
            graph_ += [[gnode_,val]]
        # prune graphs by node val:
        regraph_ = graph_reval_(graph_, [G_aves[fd] for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def sum_derHt(T, t, base_rdn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T
    derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i] + base_rdn
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    if layer[0]: sum_derH(Layer, layer)
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)

