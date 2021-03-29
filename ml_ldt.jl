using Pipe, DelimitedFiles, ScikitLearn, PyPlot

ReadFile(path) = @pipe readdlm(path, '\t') |> (_[1, :], _[2:end, :])
KeepRows(data, i, prop) = @pipe findall(x -> x == prop, data[:, i]) |> data[_, :] 
Normalize(x) = @pipe x .- minimum(x) |> _./maximum(_)

# MAIN
# ============
header, data = ReadFile("ldt.csv")

# === For ml: Fe vs Nd ===
# Reducing the rows in the data set 
data = @pipe KeepRows(data, 4, 1) |> KeepRows(_, 5, "z")
# Getting the magnet type as a binary for
t = map(x -> x == "Ferrite" ? 0 : 1, data[:, 3])
# Reducing the columns in the data set
f = convert(Array{Float64, 2}, data[:, 6:end])


# === For ml: z vs xy alignment ===
#=
# Reducing the rows in the data set 
data = KeepRows(data, 4, 1)
# Getting the orientation as a binary for
t = map(x -> x == "xy" ? 0 : 1, data[:, 5])
# Reducing the columns in the data set
f = @pipe data[:, 6:end] |> convert(Array{Float64, 2}, _)
=#

# Machine learning
#@sk_import svm: SVC
#model = SVC()

#@sk_import neighbors: KNeighborsClassifier
#model = KNeighborsClassifier()

@sk_import ensemble: RandomForestClassifier
model = RandomForestClassifier()

fit!(model, Normalize(f), t)
[t predict(model, f)]

# Plotting
#=
h = header[6:end]
i0, i1 = findall(x -> x == 0, t), findall(x -> x == 1, t)

for i=1:2:length(h)-1
  x, y = f[:, i], f[:, i+1]
  hax = [h[i], h[i+1]]
  x0, y0 = f[i0, i], f[i0, i+1] 
  x1, y1= f[i1, i], f[i1, i+1] 
  figure()

  scatter(x0, y0, color="red", label="Fe")
  scatter(x1, y1, color="green", label="Nd")
  xlabel(hax[1])
  ylabel(hax[2])
  legend()
  grid("on")

  #=
  xx0 = collect(1:length(i0))
  xx1 = collect(1:length(i1)) .+ length(i0)
  subplot(2,1,1)
  scatter(xx0, x0, color="red", label="Fe")
  scatter(xx1, x1, color="green", label="Nd")
  ylabel(hax[1])
  legend()
  grid("on")

  subplot(2,1,2)
  scatter(xx0, y0, color="red", label="Fe")
  scatter(xx1, y1, color="green", label="Nd")
  legend()
  ylabel(hax[2])
  grid("on")
  =#

  savefig("jakob_"*string(i)*".png")
tight_layout()
show()
=#
