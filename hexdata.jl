using Pipe, Contour, ImageFiltering, Statistics, Plots
import JSON, JLD
# GET DATA FROM FILE
# ====================

# Gets amr and hall vectors from a .hex file
# fname : filename, ps : package size, ss : segment size
function get_data(fname)
  bts = read(open(fname))
  ss, ps = 101, 4+9*101
  a, h = Float64[], Float64[]
  for p∈split_arr(bts, 5, ps) # package
    for s∈split_arr(p, 5, ss) # segement
      append!(a, calc_amr(s))
      append!(h, calc_hall(s))
    end
  end
  return [a, h]
end

# gets the hall shift i x axis by reading the json file 
hall_shift(path) = @pipe (path
  |> split(_, ".")[1]*".json"
  |> read_json(_)
  |> _["NumberOfSamplesToShiftHallSensorData"]
  |> Int(round(_))
)
# Reads a JSON file
read_json(fname) =
  @pipe fname |> open |> read |> String |> JSON.parse


# HANDLING OF THE HEX FILE
# =======================

# splits an array to smaller arrays 
# if the length of the array doesn't fit th new size
# the end of the array is removed
# x : array, i : start index, λ : new arrays length 
function split_arr(x, i, λ)
  sx = size(x, 1)
  mx = sx-i+1 - (sx-i+1)%λ
  [x[k:k+λ-1] for k∈i:λ:mx]
end

# Calculates the amr values
# bts : The byte array 
calc_amr(bts) = [
  @pipe (bts[k:k+1]
    |> reverse
    |> reinterpret(UInt16, _)
    |> _[1]
    |> (_ - 32768.0)/40.96
  ) for k ∈ 4:2:51]

# 2-complement operation
# n : the number, base the number base
function TwoComplement(n; base=16)
  n, m = Int(n), 2^(base - 1)
  n÷m == 0 && return n
  n%m - m
end

# Calculates the hall values
# bts : The byte array 
calc_hall(bts) = [
  @pipe (bts[k:k+1]
    |> reverse
    |> reinterpret(UInt16, _)
    |> _[1]
    |> TwoComplement
    |> _/0.91
  ) for k ∈ 52:2:99]

# IMAGE
# =========================
# Converts amr and hall vectors to images
function convert_to_images(X, fname)
  [[@pipe (X[1][i:3:end]
    |> reverse
    |> reshape(_, 72, :) 
  )
    for i ∈ [2, 1, 3]],
  [@pipe (-X[2][i:3:end] 
    |> reverse
    |> reshape(_, 72, :) 
    |> _[:, hall_shift(fname)+1:end ]
   )
   for i ∈ [2, 3, 1]]]
end

# Sets all rows to broken sensors to the average of neighbours
# A broken sensor has standard devation < 1e-12
# A, H : The amr and hall images
function fix_broken_sensors(x)
  rows = findall(x -> x < 1e-12, vec(std(x, dims=2)))
  for row ∈ sort(unique(clamp.(rows, 2, size(x, 1)-1)))
    x[row, :] = x[row-1, :]
  end
  x
end

function render(img, λ)
  for i∈1:size(img, 1), j∈1:size(img[1], 1)
    img[i][j] = imfilter(img[i][j], Kernel.gaussian(λ))
  end
  return img
end


# Makes join images of AMR and HALL
# img : AMR and HALL images
function img_join(img)
  x = [zeros(size(img[2][1])) for _∈1:3]
  for i∈1:3, j∈eachindex(img[2][i])
    a, h = img[1][i][j], img[2][i][j]
    x[i][j] = abs(h) > 500 ? h : a
  end
  return push!(img, [_x for _x∈x])
end


# PLOTTING
# ================
function grplot(img)
  norm(x) = (x .- minimum(x))/(maximum(x) .- minimum(x))
  img = vcat([im for im∈img]...)
  c = [plot(Gray.(norm(im)), ticks=nothing) for im∈img]
  plot(c..., layout=size(img, 1), ticks=nothing, 
    axis_ratio=:none, framestyle=:none, size=(700, 450))
end

# META
# ================
# calculates images from hex data
magim(fname, r) = 
  @pipe (fname
    |> get_data
    |> convert_to_images(_, fname)
    |> fix_broken_sensors
    |> render(_, r)
    |> img_join
    |> render(_, r)
  )

magim2(fname) = @pipe get_data(fname) |> convert_to_images(_, fname) |> img_join

# SPECIALS
# ================
# calculates the .hex files to matrices and saves to png plot
function selected_to_png()
  rdir = "/home/tomas/repos/susmag2/data/Loudspeakers_sorted_magplots/"
  pdir = "magplot/"
  for i∈1:11 
    dir = rdir*"s"*string(i)*"/"
    isdir(dir*pdir) || mkdir(dir*pdir)
    for f∈readdir(dir) 
      if f[end-3:end] == ".hex"
        println(f)
        @pipe (dir*f 
          |> magim(_, 2)
          |> grplot 
          |> savefig(_, dir*pdir*f[1:end-3]*".png")
        )
      end
    end
  end
end

# Finds all hex files in the directory and make images
# dir : the path to the directory
function all_to_png(dir) 
  for f∈readdir(dir) 
    if split(f, ".")[end] == "hex"
      pf = split(f, ".")[1]*"_mag.png"
      @pipe (dir*f
       |> magim(_, 2) 
       |> grplot 
       |> savefig(_, dir*pf)
      )
    end
  end
end

# change the file names so all files from a single measurement has
# the same file name
function change_names(dir)
  for f∈readdir(dir)
    f[end-3:end] == ".hex" && mv(dir*f, dir*f[1:end-16]*".hex")
    f[end-3:end] == ".txt" && mv(dir*f, dir*f[1:end-12]*".txt")
  end
end

# MAIN
# ================
#=
for fname∈readdir("systemdata")
  fname[end-3:end] != ".hex" && continue
  println("working with ", fname)
  img = magim("systemdata/"*fname, 3)
  sname = fname[1:end-4]*".jld"
  println(sname)
  JLD.save("systemdata/"*sname, "img", img)
end
=#
