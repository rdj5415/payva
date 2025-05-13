import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UiState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  notifications: {
    id: string;
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  }[];
}

const initialState: UiState = {
  sidebarOpen: true,
  theme: 'light',
  notifications: [],
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state: UiState) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setTheme: (state: UiState, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
    },
    addNotification: (state: UiState, action: PayloadAction<Omit<UiState['notifications'][0], 'id'>>) => {
      state.notifications.push({
        ...action.payload,
        id: Date.now().toString(),
      });
    },
    removeNotification: (state: UiState, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(
        (notification: UiState['notifications'][0]) => notification.id !== action.payload
      );
    },
  },
});

export const { toggleSidebar, setTheme, addNotification, removeNotification } = uiSlice.actions;
export default uiSlice.reducer; 