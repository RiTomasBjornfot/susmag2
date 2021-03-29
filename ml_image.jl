using Pipe, ScikitLearn, DelimitedFiles, JLD

# Covert data to: area, field/area and r1/r2 from main.jl
DataToFeatures(x) = @pipe (x
  |> [[_[:, j] _[:, j+1]./_[:, j] _[:, 2+j]./_[:, 3+j]] for j=1:4:size(_, 2)]
  |> hcat(_...)
  |> map(x -> isnan(x) ? 0 : x, _)
  |> _./sum(_, dims=1, init=1e-32)
)

# Features
data = JLD.load("feat.jld")["feat"]
f = DataToFeatures(data[2])

# Targets from main.jl
id = map(x -> split(x[1], ".")[1], data[1])
t0 = readdlm("result/targets.csv", ';')
# sorting the targets to the features
y = @pipe [findfirst(x -> x == id[i], t0[:, 1]) for i∈1:size(id, 1)] |> t0[_, 2]

# Machine learning
@sk_import svm: SVC
model = SVC()
fit!(model, f, y)
[y predict(model, f)]
