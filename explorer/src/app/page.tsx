import Link from "next/link";
export default function Home() {
 return <main className="mx-auto max-w-5xl px-6 py-16 text-neutral-100">
  <p className="text-blue-400 mb-3">Pipeline · Tutorial system design</p>
  <h1 className="text-4xl font-semibold mb-5">Contextual unit design space</h1>
  <p className="text-neutral-400 mb-4">Construct and compare candidate organizations of a tutorial corpus using contextual units (CUs) and canonical values. Explore the refinement graph, inspect a candidate, and copy its CU IDs to configure TaskTailor.</p>
  <p className="text-neutral-400 mb-10">The Pipeline prepares knowledge items, discovers contextual units, and produces their annotations. The design-space explorer compares prepared candidates without model calls. See the <a className="text-blue-400 underline" href="https://github.com/kixlab/starlab-contexts-pipeline#use-your-own-corpus">corpus setup guide</a>.</p>
  <Link className="block rounded-xl border border-white/15 p-6 hover:border-blue-400" href="/space"><h2 className="text-xl font-semibold mb-2">Explore design space</h2><p className="text-neutral-400">Choose a tutorial corpus, press Build, then inspect a candidate organization.</p></Link>
 </main>;
}
