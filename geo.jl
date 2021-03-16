using Pipe, JLD, PyCall
plt = pyimport("matplotlib.pyplot")
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

# TEST FUNCTIONS
# ===========================
function inpoly_point_test(pt)
  z = load("cnt.jld")["cnt"]
  #poly = z[1][1][1:end-1, :]
  poly = z[1][1]

  plt.figure("inpoly_random_test")
  plt.plot(poly[:, 1], poly[:, 2])

  println(inpoly(poly, pt), " : ", pt)
  
  c = inpoly(poly, pt) ? "green" : "red" 
  plt.plot(pt[1], pt[2], "o", color=c)
  plt.grid()
  plt.show()
end

function inpoly_random_test(iter)
  mx, mn = maximum, minimum
  rnd(p, i) = @pipe rand()*(mx(p[:, i]) - mn(p[:, i])) |> _+mn(p[:, i])

  z = load("cnt.jld")["cnt"]
  poly = z[1][1]
  plt.figure("inpoly_random_test")
  plt.plot(poly[:, 1], poly[:, 2])

  for i∈1:iter
    pt = [rnd(poly, 1) rnd(poly, 2)]
    println(inpoly(poly, pt), " : ", pt)
    c = inpoly(poly, pt) ? "green" : "red" 
    plt.plot(pt[1], pt[2], "o", color=c)
  end
  plt.grid()
  plt.show()
end
