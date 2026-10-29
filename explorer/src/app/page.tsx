import Link from "next/link";
export default function Home() {
 return <main className="mx-auto max-w-5xl px-6 py-16 text-neutral-100">
  <p className="text-blue-400 mb-3">Context Information Maps</p>
  <h1 className="text-4xl font-semibold mb-5">CIM Pipeline Explorer</h1>
  <p className="text-neutral-400 mb-10">Start with four synthetic paper-folding tutorials. The included dataset and video work locally without accounts or API keys. Use the dataset selector to open your pipeline output.</p>
  <div className="grid gap-5 md:grid-cols-2"><Link className="block rounded-xl border border-white/15 p-6 hover:border-blue-400" href="/inspect?dataset=Synthetic%20paper%20folding"><h2 className="text-xl font-semibold mb-2">Inspect knowledge</h2><p className="text-neutral-400">Inspect knowledge units, their sources, facet labels, and coverage.</p></Link><Link className="block rounded-xl border border-white/15 p-6 hover:border-blue-400" href="/space?dataset=Synthetic%20paper%20folding"><h2 className="text-xl font-semibold mb-2">Explore design space</h2><p className="text-neutral-400">Build and inspect the refinement graph of contextual factor combinations.</p></Link></div>
 </main>;
}
