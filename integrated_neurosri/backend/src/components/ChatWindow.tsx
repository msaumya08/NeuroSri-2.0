"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input" 
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Send, User, Download } from "lucide-react"

export default function MentalHealthChatbot() {
  const [messages, setMessages] = useState<Array<{id: string, role: string, content: string}>>([])
  const [input, setInput] = useState("")
  const [emotion, setEmotion] = useState({
    name: "Analyzing...",
    relaxed_confidence: 0,
    stressed_confidence: 0
  })
  const [isDownloading, setIsDownloading] = useState(false)
  const ws = useRef<WebSocket | null>(null)

  // connect to WebSocket for emotion updates - this is where the magic happens!
  useEffect(() => {
    if (typeof window !== 'undefined') {  // Check if we're in the browser
      ws.current = new WebSocket('ws://localhost:5000/ws/emotions')
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data)
        setEmotion({
          name: data.emotion,
          relaxed_confidence: data.relaxed_confidence,
          stressed_confidence: data.stressed_confidence
        })
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      // cleanup when component unmounts - we don't want memory leaks!
      return () => {
        ws.current?.close()
      }
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    // Add user message to the chat - show it immediately for better UX
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input
    }
    setMessages(prev => [...prev, userMessage])
    setInput("")

    try {
      // fire off the message to our backend - fingers crossed it works!
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input })
      })
      
      const data = await response.json()
      
      // got a response from the bot - let's add it to the chat
      const botMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response
      }
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      // Something went wrong - let the user know
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: 'Error: Could not send message. Please try again.'
      }])
    }
  }

  const handleDownloadChatHistory = async () => {
    if (messages.length === 0) {
      alert("No chat history to download.")
      return
    }

    // show loading state while we wait for the download
    setIsDownloading(true) 

    try {
      // call our fancy new API endpoint for downloading chats
      const response = await fetch('/api/chat/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_type: 'pdf' })
      })
      
      const data = await response.json()
      
      if (data.success && data.data && data.data.url) {
        // sweet! we got a URL - open it in a new tab
        window.open(data.data.url, '_blank')
      } else {
        alert(`Failed to download chat history: ${data.error || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Error downloading chat history:', error)
      alert('Error downloading chat history. Please try again.')
    } finally {
      // always reset the loading state, even if there was an error
      setIsDownloading(false)
    }
  }

  return (
    <div className="flex flex-col md:flex-row h-screen bg-background text-foreground">
      {/* Chat Interface - this is where the conversation happens */}
      <div className="w-full md:w-1/2 p-4">
        <Card className="h-full border-border bg-card">
          <CardHeader className="border-b border-border">
            <CardTitle className="text-2xl font-bold text-center text-card-foreground">Mental Health Counselor</CardTitle>
          </CardHeader>
          <CardContent className="h-[calc(100vh-200px)] overflow-y-auto p-4 space-y-4">
            {messages.map((m) => (
              <div key={m.id} className={`mb-4 ${m.role === "user" ? "text-right" : "text-left"}`}>
                <div
                  className={`inline-block p-3 rounded-lg ${
                    m.role === "user" 
                      ? "bg-primary text-primary-foreground" 
                      : "bg-secondary text-secondary-foreground"
                  }`}
                >
                  {m.role === "user" ? (
                    <User className="inline mr-2" size={18} />
                  ) : (
                    <Brain className="inline mr-2" size={18} />
                  )}
                  {m.content}
                </div>
              </div>
            ))}
          </CardContent>
          <CardFooter className="border-t border-border p-4">
            <form onSubmit={handleSubmit} className="flex w-full space-x-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-grow bg-input text-foreground border-border"
              />
              <Button type="submit" className="bg-primary text-primary-foreground hover:bg-accent">
                <Send className="w-4 h-4" />
              </Button>
              <Button 
                type="button" 
                className="bg-secondary text-secondary-foreground hover:bg-accent"
                onClick={handleDownloadChatHistory}
                disabled={isDownloading || messages.length === 0}
              >
                <Download className="w-4 h-4" />
              </Button>
            </form>
          </CardFooter>
        </Card>
      </div>

      {/* EEG and Emotion Display - the brain activity visualization */}
      <div className="w-full md:w-1/2 p-4">
        <Card className="h-full border-border bg-card">
          <CardHeader className="border-b border-border">
            <CardTitle className="text-2xl font-bold text-center text-card-foreground">Brain Activity</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center h-[calc(100vh-200px)]">
            <div className="w-full h-64 bg-muted rounded-lg overflow-hidden">
              <div className="eeg-wave"></div>
            </div>
            <div className="mt-8 text-center">
              <h2 className="text-xl font-semibold mb-2 text-card-foreground">Current Emotional </h2>
              <div className="text-3xl font-bold text-primary mb-4">
                {emotion.name.toUpperCase()}
              </div>
              <div className="flex flex-col gap-2">
                <div className="text-lg">
                  <span className="text-green-500 font-semibold">
                    Relaxed: {(emotion.relaxed_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-lg">
                  <span className="text-red-500 font-semibold">
                    Stressed: {(emotion.stressed_confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 