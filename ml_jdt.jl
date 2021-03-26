using Pipe, DelimitedFiles, ScikitLearn
KeepData(data, i, prop) = @pipe findall(x -> x == prop, data[:, i]) |> data[_, :] 
Norm(x) = @pipe x |> x./sum(_, dims=1, init=1e-32)

# main
# ============

# rading data from Jakobs file
x = readdlm("ldt.txt", '\t')
header, data = x[1, :], x[2:end, :]
data = @pipe KeepData(data, 4, 1) |> KeepData(_, 5, "z")

# Reducing the rows in the dataset
t = map(x -> x == "Ferrite" ? 0 : 1, data[:, 3])
f = convert(Array{Float64, 2}, data[:, 6:end])

# Machine learning
@sk_import svm: SVC
model = SVC()
fit!(model, Norm(f), t)
[t predict(model, f)]
