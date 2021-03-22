using Pipe, Plots #, PyCall
import Statistics: mean
import ImageFiltering: imfilter, Kernel.gaussian
import Contour as cont
import JLD, JSON
#plt  = pyimport("matplotlib.pyplot")
include("hexdata.jl")
hd = Hexdata
include("geo.jl")
pyplot()

# The distance to the mean for all pixels.
meanDistance(x) = @pipe x .- mean(x) |> _.^2 |> _./length(_) |> sqrt.(_)
# Normalizes an image.
normalize(x) = @pipe x .- minimum(x) |> _./maximum(_)
# Makes a binary image according to a limit (λ)
bim(x, λ) = @pipe meanDistance(x) |> normalize(_) |> (_ .> λ)
# Finds the min box that encloses all 1's as: (mincol, maxcol) (minrow, maxrow)
box(x) = [@pipe sum(x, dims=i) |> findall(x -> x != 0, vec(_)) |> extrema for i∈1:2]
# Render an image
render(x, λ) = imfilter(x, gaussian(λ))
# Crops an image according to c (col limits) and r (row limits)
crop(x, cr) = x[cr[1]:cr[2],cr[3]:cr[4]]
# Finds the enclosing box for x,y,z images
function minrect(X, λ)
  mn, mx = minimum, maximum
  A = [bim(x, λ) |> box for x∈X]
  c1 = mn([α[1][1] for α∈A])
  c2 = mx([α[1][2] for α∈A])
  r1 = mn([α[2][1] for α∈A])
  r2 = mx([α[2][2] for α∈A])
  (c1, c2), (r1, r2)
end

# Finds contours in an image
function findContours(x)
  cnt, sz = [], size(x)
  for cl ∈ cont.levels(cont.contours(1:sz[1], 1:sz[2], x, 1))
    for line ∈ cont.lines(cl)
      α, β  = cont.coordinates(line)
      push!(cnt, [α β])
    end
  end
  cnt
end

# Remove open contours
function removeOpen(cnt)
  x = []
  for i∈1:size(cnt, 1)
    sum(cnt[i][1, :] .- cnt[i][end, :]) != 0 && continue
    push!(x, cnt[i]) 
  end
  x
end

# Gets the contours from the image
contours(x, λ) = @pipe x |> render(_, λ) |> findContours |> removeOpen
#
# Checks if contour β lies inside α
isInside(α, β) = inpoly(α, [β[1, 1] β[1, 2]])
# Calculate min and max width for the contour.
function contourDims(c)
  rot(x, α) = x*[cos(α) -sin(α); sin(α) cos(α)]'
  extr(x, α) = @pipe rot(x, α) |> _[:, 1] |> extrema |> _[2] - _[1]
  θ, mn, mx = π/180, 1000, -1000
  for i∈1:180
    z = extr(c, i*θ)
    z > mx && (mx = z)
    z < mn && (mn = z)
  end
  mn, mx
end

# BAD!
function outerContours(cnt)
  X, outer = Bool[], []
  for i∈1:length(cnt)
    x = []
    for j∈1:length(cnt)
      i == j && continue
      append!(x, isInside(cnt[j], cnt[i]))
    end
    append!(X, !any(x))
  end
  for i∈1:size(X, 1)
    X[i] && push!(outer, cnt[i])
  end
  outer
end

# give each contour an index.
# If the index is 0, the contour is an outer
# else, it's an inner that lies inside the contour
# with the given index
function contourIndex(cnt)
  sz = length(cnt)
  index = zeros(Int, sz) 
  for i∈1:sz
    for j∈1:sz
      i == j && continue
      isInside(cnt[j], cnt[i]) && (index[i] = j)
    end
  end
  index
end

# Calculates the areas for contours in pixels
# Returns the areas and contours sorted according to area size
contourArea(c) = @pipe (c 
  |> pixelsInside 
  |> length
)

# Calculates the sum of the B field inside a contour
# Returns the areas and contours sorted according to area size
contourBfield(c) = @pipe (c
  |> pixelsInside
  |> [c[i[1], i[2]] for i∈_]
  |> sum
)

# Sorts a, b according to a
sortTwo(a, b) = @pipe sortperm(a) |> (a[_], b[_])

# Find all integer positions inside the contour
function pixelsInside(c)
  # the bounding rectangle
  mi = [@pipe c[:, i] |> extrema |> [floor(Int, _[1]) ceil(Int, _[2])] for i∈1:2]
  # inside polygon
  idx = []
  for i∈mi[1][1]:mi[1][2], j∈mi[2][1]:mi[2][2]
    inpoly(c, [i j]) && push!(idx, (i, j))
  end
  idx
end

function magnetData(fname, img)
  magnets = []
  # the raw data
  c = contours(img, 1) 
  cIndex = contourIndex(c)
  a = contourArea.(c)
  r = contourDims.(c)
  outerIndex = findall(x -> x == 0, cIndex)
  # looping over magnets in the image (outer contours)
  for i∈1:length(outerIndex)
    # get the outer area and radius
    oData = @pipe outerIndex[i] |> [a[_], r[_][1], r[_][2]]
    # get the inner area and radius
    innerIndex = findall(x -> x == outerIndex[i], cIndex) 
    d = [[a[j], r[j][1], r[j][2]] for j∈innerIndex]
    # merging inner inner data
    length(d) == 0 && (d = [[0.0, 0.0, 0.0]])
    iDataAll = +(d...)
    iDataAll[2:3] /= length(d) 
    # makes an dictionary for the magnet 
    mag = Dict()
    mag["ImageFile"] = fname
    mag["Position"] = mean(c[outerIndex[i]], dims=1) |> vec
    mag["OuterContour"] = oData
    mag["InnerContour"] = iDataAll
    # 
    push!(magnets, mag)
  end
  magnets
end

# MAIN
# =============================
rdir = "data/oriented/"
cl = 2
println("\nroot directory: ", rdir)
magnets = []
for fname∈readdir(rdir)
  fname[end-3:end] != ".hex" && continue
  f = fname[1:end-4]
  
  # convert the x, y, z images to r, z
  img = @pipe (rdir*fname
    |> hd.image(_)[3]
    |> [sqrt.(_[1].^2 + _[2].^2), _[3]] 
    |> hd.fix_broken.(_)
  )
  append!(magnets, magnetData(f, img[1]))
  println(display(magnets))
end

MagDict = Dict()
MagDict["Magnets"] = magnets
open("dict.json", "w") do fp
  JSON.print(fp, MagDict, 2)
end
  

  # replace by a contour?
  #c, r = minrect(render.(img, 3), 0.2)


  # check how many magnets we have in the image
  # ==================================
  # An "outer" contour is a magnet
  # JLD.save("test666.jld", "img", img)
  # break
  # look for out contours in the z image

  #=
  cr = [@pipe (c .- 1 
    |> extrema(_, dims=1) 
    |> [floor(_[1][1]), ceil(_[1][2]), floor(_[2][1]), ceil(_[2][2])]
    |> Int.(_)
   )  
  for c∈outerContours(cnt)]
  println("cr: ", cr)
  =#

  # info
  #=
  print("contourIndex: ", contourIndex(cnt))
  plt.imshow(img[1], cmap="gray")
  [plt.plot(c[:, 2] .- 1, c[:, 1] .- 1, color="C0") for c∈cnt]
  plt.show()
  # small images
  
  img1 = [crop(img[1], x) for x∈cr] 
  for i∈1:size(img1, 1) 
    plt.figure()
    plt.imshow(img1[i], cmap="gray")
    cntx = @pipe img1[i] |> contours(_, 1) 
    println("cntx: ", cntx)
    [plt.plot(c[:, 2] .- 1, c[:, 1] .- 1, color="C0") for c∈cntx]
  end
  plt.show()
  =#
  
  #=
  sz = size(img, 1)
  plt.figure(f, figsize=(10, 5))
  name = ["r-dir", "z-dir"]
  for i∈1:sz
  =#

    # calculating countour properties
    # ===============================
    #=
    cimg = crop(img[i], c, r) 
    cnt = contours(cimg, 1)
    #a, cnt = contourArea(cnt)
    a = contourArea2.(cnt)
    cdims = contourDims.(cnt)
    noc = size(cnt, 1)
    =#
    
    # XY alignment! X is 2* too wide!!!
    #=
    cnt = deepcopy(cnt_image)
    for i∈1:size(cnt, 1)
      cnt[i][:, 2] *= 0.5
    end
    =#

    #=
    # checking if ring shape
    ring = (noc == 2) ? isInside(cnt[1], cnt[2]) : false
    # calculate total area 
    ta = ring ? a[1] - a[2] : sum(a, init=0)
    # calculate 
    ring && (cdims = [cdims[1]])
    =#

    #=
    println("\n\t"*name[i])
    noc > 2 && println("\tWARNING: Too many contours. result may not be accurate!")
    println("\tis ring: ", ring)
    println("\ttotal area: ", ta)
    println("\tcontour dims (r1, r2): ", cdims)
    #println("\tradius ratio: ", cdims[1][2]/cdims[1][1])
    println()
    =#
    
    #=
    plt.subplot(sz+1, 1, i)
    plt.imshow(cimg, cmap="gray")
    [plt.plot(c[:, 2] .- 1, c[:, 1] .-1, color="C1") for c∈cnt]
    plt.grid()
    plt.show()
    =#
  #end
  #plt.subplot(3, 1, 3)
  #rdir*f*".jpg" |> plt.imread |> plt.imshow
  #plt.savefig(rdir*f*".png")
  #plt.show()
#end
#plt.show()
