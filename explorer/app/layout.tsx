import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://nef-qvf-diffusion-explorer.che55e.chatgpt.site"),
  title: "NEF–QVF Diffusion Explorer",
  description: "An interactive laboratory for the six natural exponential families with quadratic variance functions.",
  openGraph: {
    title: "NEF–QVF Diffusion Explorer",
    description: "Six families. One variance geometry.",
    images: [{ url: "/og.png", width: 1680, height: 945, alt: "NEF–QVF Diffusion Explorer" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "NEF–QVF Diffusion Explorer",
    description: "Six families. One variance geometry.",
    images: ["/og.png"],
  },
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({children}:{children:React.ReactNode}){
  return <html lang="en"><body>{children}</body></html>;
}
