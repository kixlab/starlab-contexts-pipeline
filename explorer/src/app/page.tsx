import Link from "next/link";
export default function Home() {
 return <main className="mx-auto max-w-5xl px-6 py-16 text-neutral-100">
  <p className="text-blue-400 mb-3">Context Information Maps</p>
  <h1 className="text-4xl font-semibold mb-5">CIM Pipeline Explorer</h1>
  <p className="text-neutral-400 mb-4">Compare schemas built from contextual units (CUs) in an annotated tutorial corpus. Select a graph node and copy its CU IDs into the retrieval, segmentation, or analysis app.</p>
  <p className="text-neutral-400 mb-10">Run the pipeline to extract and label a new corpus, or load an existing annotated bundle. Building a graph uses the loaded annotations and makes no model calls. See the <a className="text-blue-400 underline" href="https://github.com/kixlab/starlab-contexts-pipeline#use-your-own-corpus">corpus setup guide</a>.</p>
  <Link className="block rounded-xl border border-white/15 p-6 hover:border-blue-400" href="/space"><h2 className="text-xl font-semibold mb-2">Explore design space</h2><p className="text-neutral-400">Choose a dataset, press Build, then select a schema node.</p></Link>
 </main>;
}
