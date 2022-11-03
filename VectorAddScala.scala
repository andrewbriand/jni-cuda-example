import VectorAdd._

@main def VectorAddScala() =
  val a: Array[Int] = Array(1, 2, 3)
  val b: Array[Int] = Array(4, 5, 6)
  val result = VectorAdd.add(a, b)
  result foreach println
