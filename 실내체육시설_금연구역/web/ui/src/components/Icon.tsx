/**
 * 아이콘 — Lucide 한 세트만 쓴다 (ISC). 이모지 금지.
 *
 * 런타임 의존성 없이 Lucide 원본 path 데이터를 그대로 옮겼다.
 * stroke 1.5px / 24 그리드 / round cap·join 을 한 군데서 강제하므로
 * 화면마다 굵기가 달라지는 일이 생기지 않는다.
 */
import type { ReactNode, SVGProps } from 'react';

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, 'children'> {
  /** px. 기본 20 */
  size?: number;
}

function make(name: string, body: ReactNode) {
  function Icon({ size = 20, ...rest }: IconProps) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        focusable="false"
        {...rest}
      >
        {body}
      </svg>
    );
  }
  Icon.displayName = `Icon(${name})`;
  return Icon;
}

/* 내비게이션 */
export const MenuIcon = make(
  'menu',
  <>
    <path d="M4 6h16" />
    <path d="M4 12h16" />
    <path d="M4 18h16" />
  </>,
);

export const SlidersIcon = make(
  'sliders-horizontal',
  <>
    <path d="M21 4h-7" />
    <path d="M10 4H3" />
    <path d="M21 12h-9" />
    <path d="M8 12H3" />
    <path d="M21 20h-5" />
    <path d="M12 20H3" />
    <path d="M14 2v4" />
    <path d="M8 10v4" />
    <path d="M16 18v4" />
  </>,
);

export const ActivityIcon = make('activity', <path d="M22 12h-4l-3 9L9 3l-3 9H2" />);

export const BarChartIcon = make(
  'bar-chart-3',
  <>
    <path d="M3 3v18h18" />
    <path d="M18 17V9" />
    <path d="M13 17V5" />
    <path d="M8 17v-3" />
  </>,
);

export const MapIcon = make(
  'map',
  <>
    <path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z" />
    <path d="M15 5.764v15" />
    <path d="M9 3.236v15" />
  </>,
);

export const ArrowLeftIcon = make(
  'arrow-left',
  <>
    <path d="m12 19-7-7 7-7" />
    <path d="M19 12H5" />
  </>,
);

export const LayersIcon = make(
  'layers',
  <>
    <path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z" />
    <path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12" />
    <path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17" />
  </>,
);

export const ChevronLeftIcon = make('chevron-left', <path d="m15 18-6-6 6-6" />);

/* 상태 */
export const CheckCircleIcon = make(
  'circle-check',
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="m9 12 2 2 4-4" />
  </>,
);

export const AlertTriangleIcon = make(
  'triangle-alert',
  <>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </>,
);

export const AlertCircleIcon = make(
  'circle-alert',
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4" />
    <path d="M12 16h.01" />
  </>,
);

export const InfoIcon = make(
  'info',
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4" />
    <path d="M12 8h.01" />
  </>,
);

/* 조작 */
export const XIcon = make(
  'x',
  <>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </>,
);

export const ChevronDownIcon = make('chevron-down', <path d="m6 9 6 6 6-6" />);

export const ChevronRightIcon = make('chevron-right', <path d="m9 18 6-6-6-6" />);

export const RefreshIcon = make(
  'refresh-cw',
  <>
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </>,
);

export const LoaderIcon = make('loader-circle', <path d="M21 12a9 9 0 1 1-6.219-8.56" />);

export const DownloadIcon = make(
  'download',
  <>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="M7 10l5 5 5-5" />
    <path d="M12 15V3" />
  </>,
);

/* 콘텐츠 */
export const InboxIcon = make(
  'inbox',
  <>
    <path d="M22 12h-6l-2 3h-4l-2-3H2" />
    <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11" />
  </>,
);

export const MapPinIcon = make(
  'map-pin',
  <>
    <path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0" />
    <circle cx="12" cy="10" r="3" />
  </>,
);

export const ClockIcon = make(
  'clock',
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 6v6l4 2" />
  </>,
);

export const WalletIcon = make(
  'wallet',
  <>
    <path d="M19 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0 0 4h15a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5" />
    <path d="M18 12a2 2 0 0 0 0 4h4v-4Z" />
  </>,
);

export const UsersIcon = make(
  'users',
  <>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </>,
);

export const FlaskIcon = make(
  'flask-conical',
  <>
    <path d="M10 2v7.31a1 1 0 0 1-.14.51L4.2 19.2A2 2 0 0 0 5.9 22h12.2a2 2 0 0 0 1.7-2.8l-5.66-9.38a1 1 0 0 1-.14-.51V2" />
    <path d="M8.5 2h7" />
    <path d="M7 16h10" />
  </>,
);
