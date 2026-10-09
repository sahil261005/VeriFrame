import axios from 'axios';

// backend url, uses localhost if the env var isnt set
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
});

// add the jwt token to every request if we have one saved
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// if the token is expired or bad, log out and go to the login page
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !window.location.pathname.includes('/login') && !window.location.pathname.includes('/register')) {
      localStorage.removeItem('token');
      localStorage.removeItem('user_email');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// api calls
export const authService = {
  async login(email, password) {
    const response = await api.post('/auth/login', { email, password });
    if (response.data?.access_token) {
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('user_email', email);
    }
    return response.data;
  },

  async register(email, password) {
    const response = await api.post('/auth/register', { email, password });
    if (response.data?.access_token) {
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('user_email', email);
    }
    return response.data;
  },

  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user_email');
  },

  isAuthenticated() {
    return !!localStorage.getItem('token');
  },

  getUserEmail() {
    return localStorage.getItem('user_email') || '';
  }
};

export const analysisService = {
  async uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getAnalysis(jobId) {
    const response = await api.get(`/analysis/${jobId}`);
    return response.data;
  },

  async getPDFReport(jobId) {
    const response = await api.get(`/report/${jobId}/pdf`, {
      responseType: 'blob',
    });
    return response.data;
  },

  createEventStream(jobId, onEvent, onError) {
    const streamUrl = `${API_URL}/stream/${jobId}`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onEvent) onEvent(data);
      } catch (err) {
        console.error('Error parsing SSE event data:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE connection error:', err);
      if (onError) onError(err);
      eventSource.close();
    };

    return eventSource;
  }
};

export default api;
