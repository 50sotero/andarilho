"use client";

import dynamic from "next/dynamic";

export const ViewerClient = dynamic(() => import("./ViewerCanvas"), {
  ssr: false,
  loading: () => <p>Loading the 3D viewer...</p>,
});
