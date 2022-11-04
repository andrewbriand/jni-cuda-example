import VectorAdd._

@main def VectorAddScala() =
  val a: Array[Int] = (1 to 10000).toArray
  val b: Array[Int] = (1 to 10000).toArray
  val result = VectorAdd.add(a, b)
  (0 to 9).foreach(x => println(result(x)))
  println("...")
  (9990 to 9999).foreach(x => println(result(x)))
