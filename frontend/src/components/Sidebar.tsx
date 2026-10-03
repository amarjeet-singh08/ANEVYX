import { Activity, FileScan } from 'lucide-react';

interface SidebarItemProps {
  icon: any;
  label: string;
  active?: boolean;
  onClick: () => void;
}

function SidebarItem({ icon: Icon, label, active, onClick }: SidebarItemProps) {
  return (
    <div 
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 12px',
        borderRadius: '8px',
        cursor: 'pointer',
        background: active ? 'var(--accent-muted)' : 'transparent',
        color: active ? 'var(--accent)' : 'var(--text-muted)',
        fontWeight: active ? 600 : 500,
        transition: 'all 0.2s ease',
        whiteSpace: 'nowrap'
      }}
    >
      <Icon size={18} />
      <span style={{ fontSize: '14px' }}>{label}</span>
    </div>
  );
}

export function Sidebar({ currentTab, setCurrentTab }: { currentTab: string, setCurrentTab: (t: string) => void }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
          <Activity color="var(--accent)" />
          ANEVYX
        </h2>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Imaging Intelligence
        </div>
      </div>
      
      <div className="sidebar-items" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <SidebarItem icon={FileScan} label="Analyze" active={currentTab === 'analyze'} onClick={() => setCurrentTab('analyze')} />
      </div>
      
      <div className="sidebar-footer" style={{ marginTop: 'auto', fontSize: '12px', color: 'var(--text-muted)' }}>
        Processing: <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>LOCAL</span>
      </div>
    </div>
  );
}
