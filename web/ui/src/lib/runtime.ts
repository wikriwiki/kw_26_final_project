/** Build-time deployment mode. The API independently enforces the same rule. */
export const READ_ONLY = import.meta.env.VITE_READ_ONLY === 'true';
