"""
  line_Ps is a principal version of 1st-level 1D algorithm
  Operations:
  -
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|,
  rather than a minimum for directly defined match: albedo of an object doesn't correlate with its predictive value.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they do overlap in line_patterns_olp.
"""

using Images, ImageView, CSV

# instead of the class in Python version in Julia values are stored in the struct
mutable struct Cdert_
    i::Int
    p::Int
    d::Int
    m::Int
    mrdn::Bool
end


# mutable struct Sublayers
#     L::Int32
#     I::Int32
#     D::Int32
#     M::Int32  # summed ave - abs(d), different from D
#     Rdn::Int32  # 1 + binary dert.mrdn cnt / len(dert_)
#     x0::Int32
# end

mutable struct CP
    L::Int
    I::Int
    D::Int
    M::Int # summed ave - abs(d), different from D
    Rdn::Int  # 1 + binary dert.mrdn cnt / len(dert_)
    x0::Int
    dert_::Vector{Cdert_}  # contains (i, p, d, m, mrdn)
    subset::Any # 1st sublayer' rdn, rng, xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ dert_], 3rd: sublayers[ sub_P_[ dert_]]:
    sublayers::Vector{Any}  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
end

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)

init_y = 501  # starting row, set 1 for the whole frame, mostly not needed 
halt_y = 501  # ending row, set 999999999 for arbitrary image.
#! these values will be one more (+1) in the Julia version because of the numbering specifics
"""
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
"""

function range_incr_P_(rootP, P_, rdn, rng)
    comb_sublayers = []  #+ The same notation 
    for (id, P) in enumerate(P_)
        # if P.M - P.Rdn * ave_M * P.L > ave_M * rdn and P.L > 2  # M value adjusted for xP and higher-layers redundancy
        # if P.M - P.Rdn * ave_M * P.L > ave_M * rdn #and P.L > 2  # M value adjusted for xP and higher-layers redundancy
            """ P is Pm   
            min skipping P.L=3, actual comp rng = 2^(n+1): 1, 2, 3 -> kernel size 4, 8, 16...
            """
            rdn += 1; rng += 1
            # P.subset = rdn, rng, [],[],[],[]  # 1st sublayer params, []s: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            push!(P_[id].subset, [rdn, rng, [], [], [], []])  # 1st sublayer params, []s: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            
            sub_Pm_, sub_Pd_ = [], []  # initialize layers, concatenate by intra_P_ in form_P_
            push!(P_[id].sublayers, [sub_Pm_, sub_Pd_])  # 1st layer
            rdert_ = []
        # end
    #         _i = P.dert_[0].i
    #         for dert in P.dert_[2::2]:  # all inputs are sparse, skip odd pixels compared in prior rng: 1 skip / 1 add to maintain 2x overlap
    #             # skip predictable next dert, local ave? add rdn to higher | stronger layers:
    #             d = dert.i - _i
    #             rp = dert.p + _i  # intensity accumulated in rng
    #             rd = dert.d + d  # difference accumulated in rng
    #             rm = ave*rng - abs(rd)  # m accumulated in rng
    #             rmrdn = rm < 0
    #             rdert_.append(Cdert(i=dert.i, p=rp, d=rd, m=rm, mrdn=rmrdn))
    #             _i = dert.i
    #         sub_Pm_[:] = form_P_(P, rdert_, rdn, rng, fPd=False)  # cluster by rm sign
    #         sub_Pd_[:] = form_P_(P, rdert_, rdn, rng, fPd=True)  # cluster by rd sign
    #     if rootP and P.sublayers:
    #         new_comb_sublayers = []
    #         for (comb_sub_Pm_, comb_sub_Pd_), (sub_Pm_, sub_Pd_) in zip_longest(comb_sublayers, P.sublayers, fillvalue=([],[])):
    #             comb_sub_Pm_ += sub_Pm_  # remove brackets, they preserve index in sub_Pp root_
    #             comb_sub_Pd_ += sub_Pd_
    #             new_comb_sublayers.append((comb_sub_Pm_, comb_sub_Pd_))  # add sublayer
    #         comb_sublayers = new_comb_sublayers
    end
    # if rootP:
    #     rootP.sublayers += comb_sublayers  # no return
    # end
end

function form_P_(rootP, dert_; rdn, rng, fPd)  # after semicolon in Julia keyword args are placed
    # initialization:
    P_ = CP[]  # structure to store all the form_P (layer1) output
    x = 0
    _sign = nothing  # to initialize 1st P, (nothing != True) and (nothing != False) are both True

    for dert in dert_  # segment by sign
        if fPd == true
            sign = dert.d > 0
        else
            sign = dert.m > 0
        end

        if sign != _sign
            # sign change, initialize and append P
            L = 1; I = dert.p; D = dert.d; M = dert.m; Rdn = dert.mrdn + 1; x0 = x; sublayers = [] # Rdn starts from 1
            push!(P_, CP(L, I, D, M, Rdn, x0, [dert], [], []))  # save data in the struct
        else
            # accumulate params:
            P_[end].L += 1; P_[end].I += dert.p; P_[end].D += dert.d; P_[end].M += dert.m; P_[end].Rdn += dert.mrdn
            push!(P_[end].dert_, dert)
        end
        x += 1
        _sign = sign
    end

    """    
    due to separate aves, P may be processed by both or neither of r fork and d fork
    add separate rsublayers and dsublayers?
    """
    range_incr_P_(rootP, P_, rdn, rng)
    # deriv_incr_P_(rootP, P_, rdn, rng)
    if logging == 2
        if fPd == false
            CSV.write("./layer2_Pm_log_jl.csv", P_)
        else
            CSV.write("./layer2_Pd_log_jl.csv", P_)
        end
    end
    
    # return P_  # used only if not rootP, else packed in rootP.sublayers
end

function line_Ps_root(pixel_)  # Ps: patterns, converts frame_of_pixels to frame_of_patterns, each pattern may be nested
    dert_ = Cdert_[] # line-wide i_, p_, d_, m_, mrdn_

    _i = pixel_[1]  #! differs from the python version # cross_comparison:
    for i in pixel_[2:end]  # pixel i is compared to prior pixel _i in a row:
        d = i - _i  # accum in rng
        p = i + _i  # accum in rng
        m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
        mrdn = m < 0  # 1 if abs(d) is stronger than m, redundant here
        push!(dert_, Cdert_(i, p, d, m, mrdn))  # save data in the struct
        _i = i
    end

    # form patterns, evaluate them for rng+ and der+ sub-recursion of cross_comp:
    Pm_ = form_P_(nothing, dert_, rdn = 1, rng = 1, fPd = false)  # rootP=None, eval intra_P_ (calls form_P_)
    Pd_ = form_P_(nothing, dert_, rdn = 1, rng = 1, fPd = true)

    if logging == 1
        CSV.write("./layer1_log_jl.csv", dert_)
    end

    return [Pm_, Pd_]  # input to 2nd level
end


render = 0
fline_PPs = 0
frecursive = 0
logging = 0  # logging of local functions variables

# image_path = "./line_1D_alg/raccoon.jpg";
image_path = "/home/alex/Python/CogAlg/line_1D_alg/raccoon.jpg";
image = nothing

# check if image exist
if isfile(image_path)
    image = load(image_path)  # read as N0f8 (normed 0...1) type array of cells
else
    println("ERROR: Image not found!")
end

gray_image = Gray.(image)  # convert rgb N0f8 to gray N0f8 array of cells
img_channel_view = channelview(gray_image)  # transform to the array of N0f8 numbers
gray_image_int = convert.(Int, trunc.(img_channel_view .* 255))  # finally get array of 0...255 numbers
# imshow(gray_image)

# Main
Y, X = size(gray_image) # Y: frame height, X: frame width
frame = []

# y is index of new row pixel_, we only need one row, use init_y=1, halt_y=Y for full frame
for y = init_y:min(halt_y, Y)
    line_Ps_root(gray_image_int[y, :])  # line = [Pm_, Pd_]
end

