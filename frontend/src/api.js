import axios from 'axios';

// configure backend base url
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
});

// automatically attach JWT bearer token to requests if it exists in localstorage
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

// api service calls
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
  }
};

export default api;
