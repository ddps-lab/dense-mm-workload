name := "MLP"

version := "1.0"
scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0"
libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
