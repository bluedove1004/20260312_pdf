/** Application root with React Router and persistent navigation bar. */

import React from 'react';
import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import DocumentViewer from './components/DocumentViewer';
import SettingsPanel from './components/SettingsPanel';

const NavBar: React.FC = () => {
  const { pathname } = useLocation();

  const navLink = (to: string, label: string, icon: React.ReactNode) => (
    <Link
      to={to}
      className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200
        ${pathname === to ? 'text-brand-300 bg-brand-900/30' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'}`}
      id={`nav-${to.replace('/', '') || 'home'}`}
    >
      {icon}
      {label}
    </Link>
  );

  return (
    <nav className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900/70 backdrop-blur sticky top-0 z-50">
      <Link to="/" className="flex items-center gap-3" id="nav-logo">
        <div className="w-8 h-8 bg-gradient-to-br from-brand-500 to-purple-600 rounded-lg flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <span className="font-semibold text-gray-200 text-sm">Book Scan OCR</span>
      </Link>

      <div className="flex items-center gap-1">
        {navLink('/', '대시보드',
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
        )}
        {navLink('/settings', '설정',
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        )}
      </div>
    </nav>
  );
};

const App: React.FC = () => (
  <BrowserRouter>
    <div className="min-h-screen bg-gray-950 hero-bg flex flex-col">
      <NavBar />
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/documents/:id" element={<DocumentViewer />} />
          <Route path="/settings" element={
            <div className="p-6">
              <SettingsPanel />
            </div>
          } />
        </Routes>
      </main>
    </div>
  </BrowserRouter>
);

export default App;
