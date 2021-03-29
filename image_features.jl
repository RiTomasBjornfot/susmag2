using Pipe 
import Statistics: mean
import ImageFiltering: imfilter, Kernel.gaussian
import DelimitedFiles: writedlm
import Contour as cont
import JLD, JSON

include("hexdata.jl")
hd = Hexdata
pyplot()

# The distance to the mean for all pixels.
# meanDistance(x) = @pipe x .- mean(x) |> _.^2 |> _./length(_) |> sqrt.(_)
# Normalizes an image.
# normalize(x) = @pipe x .- minimum(x) |> _./maximum(_)
# Makes a binary image according to a limit (λ)
#bim(x, λ) = @pipe meanDistance(x) |> normalize(_) |> (_ .> λ)
# Finds the min box that encloses all 1's as: (mincol, maxcol) (minrow, maxrow)
# box(x) = [@pipe sum(x, dims=i) |> findall(x -> x != 0, vec(_)) |> extrema for i∈1:2]

# checks if a point (pt) lies in the polygon's enclosing rectangle
function inrect(poly, pt)
  dp = [poly[:, 1] .- pt[1] poly[:, 2] .- pt[2]]
  z = [
       length(findall(z -> z > 0, dp[:, 1])),
       length(findall(z -> z < 0, dp[:, 1])),
       length(findall(z -> z > 0, dp[:, 2])),
       length(findall(z -> z < 0, dp[:, 2]))
      ]
  all(x -> x != 0, z)
end

# checks is a point lies inside a polygon 
# using the winding number
inpoly(poly, pt) = @pipe (poly .- pt
    |> angle.(_*[1 im]') 
    |> vec 
    |> rad2deg.(_) 
    |> diff
    |> [abs(x) < 180 ? x : x - sign(x)*360 for x∈_]
    |> sum
    |> round
    |> abs
    |> (_ == 360)
  )

# Render an image
render(x, λ) = imfilter(x, gaussian(λ))
# Crops an image according to c (col limits) and r (row limits)
crop(x, cr) = x[cr[1]:cr[2],cr[3]:cr[4]]

# Finds contours in an image
function findContours(x)
  cnt, sz = [], size(x)
  for cl ∈ cont.levels(cont.contours(1:sz[1], 1:sz[2], x, 1))
    for line ∈ cont.lines(cl)
      α, β  = cont.coordinates(line)
      push!(cnt, [α β] .- 1)
    end
  end
  cnt
end

# Removes open contours.
function removeOpen(cnt)
  x = []
  for i∈1:size(cnt, 1)
    sum(cnt[i][1, :] .- cnt[i][end, :]) != 0 && continue
    push!(x, cnt[i]) 
  end
  x
end

# Gets the contours from the image.
contours(x, λ) = @pipe x |> render(_, λ) |> findContours |> removeOpen
# Checks if contour β lies inside α
isInside(α, β) = inpoly(α, [β[1, 1] β[1, 2]])
# Calculates min and max width for the contour.
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

# Gives each contour an index:
# If the contour is an outer: 0
# If it's an inner, it gets the index of the outer
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

# Calculates the areas for contours in pixels.
# Returns the areas and contours sorted according to area size
contourArea(c) = @pipe (c 
  |> pixelsInside 
  |> length
)

# Calculates the sum of the B field inside a contour
# Returns the areas and contours sorted according to area size
contourBfield(img, c) = @pipe (c
  |> pixelsInside
  |> [img[i[1], i[2]] for i∈_]
  |> abs.(_)
  |> sum
)

# Sorts a, b according to a
# sortTwo(a, b) = @pipe sortperm(a) |> (a[_], b[_])

# Find all integer positions inside the contour
function pixelsInside(c)
  # the bounding rectangle
  mi = [@pipe (c[:, i] 
    |> extrema 
    |> [floor(Int, _[1]) ceil(Int, _[2])])
  for i∈1:2]
  # inside polygon
  idx = []
  for i∈mi[1][1]:mi[1][2], j∈mi[2][1]:mi[2][2]
    inpoly(c, [i j]) && push!(idx, (i, j))
  end
  idx
end

# convert the x, y, z images to r, z
convertImage(img) = @pipe (img
  |> [sqrt.(_[1].^2 + _[2].^2), _[3]] 
  |> hd.fix_broken.(_)
)

# Calculate the magnet data for all images
function MagnetData(fname, img)
  robs, zobs = [], []
  for index∈1:length(img)
    # the raw data
    c = contours(img[index], 1) 
    cIndex = contourIndex(c)
    a = contourArea.(c)
    r = contourDims.(c)
    b = [contourBfield(img[index], x) for x∈c]
    oIndex = findall(x -> x == 0, cIndex)
    # looping over magnets in the image (outer contours)
    for i∈1:length(oIndex)
      # get the outer area, field and radius
      oData = @pipe oIndex[i] |> [a[_], b[_], r[_][1], r[_][2]]
      # get the inner area, field and radius
      iIndex = findall(x -> x == oIndex[i], cIndex) 
      d = [[a[j], b[j], r[j][1], r[j][2]] for j∈iIndex]
      # merging inner data
      length(d) == 0 && (d = [[0.0, 0.0, 0.0, 0.0]])
      iData = +(d...)
      iData[2:3] /= length(d) 
      # get the position of the contour
      pos = @pipe c[oIndex[i]] |> mean(_, dims=1) |> vec |> round.(_, digits=2)
      x = [pos..., oData..., iData...]
      index == 1 ? push!(robs, x) : push!(zobs, x)
    end
  end
  # merging r and z data
  info, data = Any[fname], Float64[]
  if length(robs) == 1 && length(zobs) == 1
    append!(info, robs[1][1:2]..., zobs[1][1:2]...)
    append!(data, robs[1][3:end]..., zobs[1][3:end]...)
  else
    println("More then one magnet is not supported yet!")
    println("no magnets r: ", length(robs))
    println("no magnets z: ", length(zobs))
  end
  return info, data
end

# MAIN
# =============================
rdir = "data/enuppned/"
println("\nroot directory: ", rdir)
#imageIndex = 2
info, features = Any[], []
for fname∈readdir(rdir)
  fname[end-3:end] != ".hex" && continue
  iobs, obs = @pipe (rdir*fname 
    |> hd.image(_)[3] 
    |> convertImage 
    |> MagnetData(fname, _)
  )
  push!(info, iobs)
  push!(features, obs)
end

@pipe hcat(features...)' |> convert(Array, _) |> JLD.save("data.jld", "feat", (info, _))
