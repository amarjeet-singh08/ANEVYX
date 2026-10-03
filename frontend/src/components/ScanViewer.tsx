interface ScanViewerProps {
  title: string;
  imageSrc: string | null;
  heatmapSrc?: string | null;
}

export function ScanViewer({ title, imageSrc, heatmapSrc }: ScanViewerProps) {
  if (!imageSrc) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h3 style={{ fontSize: '14px', color: 'var(--text-muted)' }}>{title}</h3>
        <div style={{ 
          flex: 1, 
          background: 'var(--surface-hover)', 
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)'
        }}>
          No scan available
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', minWidth: 0 }}>
      <h3 style={{ fontSize: '14px', color: 'var(--text-main)', fontWeight: 500 }}>{title}</h3>
      
      <div style={{ 
        flex: 1, 
        background: '#000', 
        borderRadius: 'var(--radius-md)',
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{ 
          position: 'relative', 
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <img 
            src={imageSrc} 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '100%', 
              objectFit: 'contain'
            }} 
            alt="Scan" 
          />
          {heatmapSrc && (
              <img 
              src={heatmapSrc} 
              style={{ 
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%', 
                height: '100%', 
                objectFit: 'contain',
                pointerEvents: 'none'
              }} 
              alt="Heatmap overlay" 
            />
          )}
        </div>
      </div>
    </div>
  );
}
