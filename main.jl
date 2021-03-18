using Pipe, Statistics, PyCall, ImageFiltering, Plots, Contour, Images
plt  = pyimport("matplotlib.pyplot")
include("hexdata.jl")
include("geo.jl")

# flatten an array of arrays
flatten(x) = collect(Iterators.flatten(x))
# The distance to the mean for all pixels.
mean_distance(x) = @pipe x .- mean(x) |> _.^2 |> _./length(_) |> sqrt.(_)
# Normalizes an image.
normalize(x) = @pipe x .- minimum(x) |> _./maximum(_)
# Makes a binary image according to a limit (λ)
bim(x, λ) = @pipe mean_distance(x) |> normalize(_) |> (_ .> λ)
# Finds the min box that encloses all 1's as: (mincol, maxcol) (minrow, maxrow)
box(x) = [@pipe sum(x, dims=i) |> findall(x -> x != 0, vec(_)) |> extrema for i∈1:2]
# render an image
render(x, λ) = imfilter(x, Kernel.gaussian(λ))
# finds the enclosing box for x,y,z images
function minrect(X, λ)
  mn, mx = minimum, maximum
  A = [bim(x, λ) |> box for x∈X]
  c1 = mn([α[1][1] for α∈A])
  c2 = mx([α[1][2] for α∈A])
  r1 = mn([α[2][1] for α∈A])
  r2 = mx([α[2][2] for α∈A])
  (c1, c2), (r1, r2)
end
# Crops an image according to c (col limits) and r (row limits)
crop(x, c, r) = x[r[1]:r[2],c[1]:c[2]]

# convert cartesian index to array
cart_to_arr(x) = [x[i] for i∈1:length(x)]

# Contours
# ================

# finds contours in an image (img). Number of levels = ls
function findContours(img, ls)
  x, y, z = [], [], []
  for cl ∈ levels(contours(1:size(img, 1), 1:size(img, 2), img, ls))
    for line ∈ lines(cl)
      append!(z, level(cl))
      α, β  = coordinates(line)
      push!(x, α)
      push!(y, β)
    end
  end
  [[[x[i] y[i]] for i ∈ 1:size(x, 1)], z]
end

# remove open contours
function remove_open(cnt)
  α, β = [], []
  for i ∈ 1:size(cnt[1], 1)
    x, z = cnt[1][i], cnt[2][i]
    sum(x[1, :] .- x[end, :]) != 0 && continue
    push!(α, x) 
    append!(β, z)
  end
  [α, β]
end


# keeps only contours with levels: lvls
function keep_levels(cnt, lvls)
  ncnt = [[], []]
  for lvl∈lvls
    for i∈1:size(cnt[1], 1)
      if cnt[2][i] == lvl
        push!(ncnt[1], cnt[1][i])
        append!(ncnt[2], lvl)
      end
    end
  end
  ncnt
end

# keeps only contours without a contour inside
function keep_inner(cnt)
  size(cnt[1], 1) == 1 && (return cnt)
  mp = [[c[1, 1] c[1, 2]] for c∈cnt[1]]
  ncnt = [[], []]
  for i∈1:size(cnt[1], 1)
    t = []
    for j∈1:size(cnt[1], 1)
      if i != j
        append!(t, inpoly(cnt[1][i], mp[j]))
      end
    end
    if all(x -> x == false, t)
      push!(ncnt[1], cnt[1][i])
      append!(ncnt[2], cnt[2][i])
    end
  end
  ncnt
end

# This is the function that get the important contours
# OLD
feature_contours(img, cl) = @pipe (img
  |> render(_, 1) 
  |> normalize 
  |> findContours(_, cl - 1)
  |> remove_open
  |> keep_levels(_, [1/cl (cl - 1)/cl])
  |> keep_inner
)

# gets the contours from the image
function mag_contours(img, cl) 
  cnt = @pipe (img
    |> render(_, 1) 
    |> findContours(_, cl - 1)
    |> remove_open
  )
  return cnt
end


# checks if contour c1 lies inside c2
is_inside(c1, c2) = inpoly(c1, [c2[1, 1] c2[1, 2]])
# checks is contour c1 and c2 forms a dougnut shape
is_dougnut(c1, c2) = is_inside(c1, c2)
# calculate min and max width for the contour.
function contour_dims(c)
  # rotates x, α degree in the plane
  rot(x, α) = x*[cos(α) -sin(α); sin(α) cos(α)]'
  θ = π/180
  mn, mx = 10000, -1000
  for i∈1:360
    z = @pipe rot(c, i*θ) |> _[:, 1] |> extrema |> _[2] - _[1]
    z > mx && (mx = z)
    z < mn && (mn = z)
  end
  mn, mx
end

# calculates the area for a contour in pixels
# returns the area sorted
function contour_area(c) 
  a = [length(pixels_inside(x)) for x∈c]
  sort(a, rev=true), c[sortperm(a, rev=true)] 
end
# find all integer positions inside the contour
function pixels_inside(c)
  # inside bounding rectangle
  mi = [@pipe c[:, i] |> extrema |> [floor(Int, _[1]) ceil(Int, _[2])] for i∈1:2]
  # inside polygon
  idx = []
  for i∈mi[1][1]:mi[1][2], j∈mi[2][1]:mi[2][2]
    inpoly(c, [i j]) && push!(idx, (i, j))
  end
  idx
end

# gets the size and average value inside the contour
# cnt : an array of contours
# old
contour_props(cnt, img) = 
[@pipe (cnt[1][i] 
    |> pixels_inside
    |> [img[i...] for i∈_] 
    |> (length(_), mean(_))
  )
 for i∈1:length(cnt[1])]

# MAIN
# =============================

rdir = "data/enupp/"
cl = 2
println("\nroot dir: ", rdir)
println("levels: ", [1/cl (cl - 1)/cl])

for fname∈readdir(rdir)
  fname[end-3:end] != ".hex" && continue
  f = fname[1:end-4]
  println("\n"*f)
  path = rdir*fname
  img_all = magim2(rdir*fname)
  img = @pipe (img_all[3] 
    |> [sqrt.(_[1].^2 + _[2].^2), _[3]] 
    |> fix_broken_sensors.(_)
  )
  
  c, r = minrect(render.(img, 3), 0.2)
  sz = size(img, 1)
  plt.figure(f, figsize=(10, 5))
  name = ["r-dir", "z-dir"]
  for i∈1:sz
    cimg = crop(img[i], c, r) 
    cnt = mag_contours(cimg, cl)[1] 
    
    #=
    # XY alignment! X is 2* too wide!!!
    for i∈1:size(cnt, 1)
      cnt[i][:, 2] *= 0.5
    end
    =#

    a, cnt = contour_area(cnt)
    noc = size(cnt, 1)
    cdims = contour_dims.(cnt)
    
    # checking if dougnut
    dnut = (noc == 2) ? is_dougnut(cnt[1], cnt[2]) : false
    # calculate total area 
    ta = dnut ? -(a...) : +(a...)
    # calculate 
    dnut && (cdims = [cdims[1]])
    
    println("\n\t"*name[i])
    #println("\tnumber of contours: ", noc)
    noc > 2 && println("\tWARNING: Too many contours. result may not be accurate!")
    println("\tdoungnut: ", dnut)
    #println("\tcontour areas: ", a)
    println("\ttotal area: ", ta)
    println("\tcontour dims (r1, r2): ", cdims)
    println("rr: ", cdims[1][2]/cdims[1][1])
    data = [Int(dnut), ta, cdims[1]] #, Int(cdims[1][1]), Int(cdims[1][2])]
    print("\t=> ")
    [print(string(d)*" ") for d∈data]
    println()


    #=
    p = contour_props(cnt, cimg)
    println("\n\t"*name[i])
    println("\tsize: ", [pp[1] for pp∈p])
    println("\tdensity: ", [pp[2] |> round |> Int for pp∈p])
    println("\tenergy: ", [*(pp[1:2]...) |> round |> Int for pp∈p])
    =#
    
    plt.subplot(1,sz,i)
    plt.imshow(cimg, cmap="gray")
    [plt.plot(c[:, 2] .- 1, c[:, 1] .-1, color="C1") for c∈cnt]
    plt.grid()
  end
  rdir*f*".jpg" |> load |> plot |> display
  plt.savefig(rdir*f*".png")
  plt.close()
end
