using Pipe, Statistics, PyCall, ImageFiltering, Plots, Contour, Images
plt  = pyimport("matplotlib.pyplot")
include("hexdata.jl")

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
      #println(level(cl))
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

# keeping only levels defined in sl
function keep_levels(cnt, lvls)
  [@pipe findall(x -> x == lvl, cnt[2]) |> cnt[1][_]
    for lvl∈lvls]
end
# MAIN
# =============================
rdir = "orienteddata/"
for fname∈readdir(rdir)
  fname[end-3:end] != ".hex" && continue
  path = rdir*fname
  imgall = magim2(path)
  img = @pipe (imgall[3] 
    |> [sqrt.(_[1].^2 + _[2].^2), _[3]] 
    |> fix_broken_sensors.(_)
  )
  
  # plottning org data
  #=
  for (i, name) ∈ enumerate(["AMR", "HALL", "JOIN"])
    plt.figure(name)
    for img ∈ imgall[i]
      plt.imshow(img, cmap="gray")
    end
  end
  =#

  c, r = minrect(render.(img, 3), 0.2)
  println(c, r)
  sz = size(img, 1)
  plt.figure(fname[1:end-4], figsize=(15, 10))
  for i∈1:sz
    simg = @pipe img[i] |> crop(_, c, r) 
    cnt = @pipe (simg 
      |> render(_, 3) 
      |> normalize 
      |> findContours(_, 4)
      |> remove_open
      |> keep_levels(_, [0.2, 0.8])
    )
    plt.subplot(2,sz,i)
    plt.imshow(simg, cmap="gray")
    # min contours
    [plt.plot(c[:, 2].- 1, c[:, 1].- 1, color="C1") for c∈cnt[1]]
    # max contours
    [plt.plot(c[:, 2].- 1, c[:, 1].- 1, color="C0") for c∈cnt[2]]
    
    #JLD.save(fname[1:end-4]*".jld", "cnt", cnt) 
    
    plt.subplot(2,sz,i+sz)
    plt.hist(vec(simg), 50)
    plt.grid()

  end
  rimg = load(rdir*fname[1:end-4]*".jpg")
  println(fname[1:end-4])
  display(plot(rimg))
  #plt.savefig(rdir*fname[1:end-4]*".png")
  plt.show()
end
