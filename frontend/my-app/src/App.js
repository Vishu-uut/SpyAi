import React, { useEffect, useState } from "react";
import io from "socket.io-client";

const API_URL = "http://localhost:5000";
const socket = io(API_URL);

function App() {
    const [videos, setVideos] = useState([]);

    useEffect(() => {
        fetchVideos();

        socket.on("new_video", (data) => {
            setVideos((prev) => [...prev, data.filename]);
        });

        socket.on("remove_video", (data) => {
            setVideos((prev) => prev.filter(video => video !== data.filename));
        });

        return () => {
            socket.off("new_video");
            socket.off("remove_video");
        };
    }, []);

    const fetchVideos = async () => {
        try {
            const response = await fetch(`${API_URL}/videos`);
            if (!response.ok) throw new Error("Network response was not ok");
            const videoList = await response.json();
            setVideos(videoList);
        } catch (error) {
            console.error("Error fetching videos:", error);
        }
    };

    const handleAction = async (action, filename) => {
        await fetch(`${API_URL}/action/${action}/${filename}`, { method: "POST" });

        if (action === "deny") {
            setVideos(videos.filter(video => video !== filename)); // Remove from UI
        }
    };

    return (
        <div style={{ padding: 20 }}>
            <h2>Live Video Feed</h2>
            {videos.length === 0 && <p>No videos found</p>}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                {videos.map((video, index) => (
                    <div key={index} style={{ border: "1px solid #ccc", padding: 10 }}>
                        <video width="100%" controls>
                            <source src={`${API_URL}/videos/${video}`} type="video/mp4" />
                        </video>
                        <div style={{ marginTop: 10 }}>
                            <button onClick={() => handleAction("alert", video)} style={{ marginRight: 5, background: "red", color: "white" }}>
                                Alert
                            </button>
                            <button onClick={() => handleAction("deny", video)} style={{ background: "green", color: "white" }}>
                                Deny
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;
