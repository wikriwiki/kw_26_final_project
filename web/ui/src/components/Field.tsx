/**
 * 입력 — 스펙 §8.
 * - 레이블은 항상 보인다. placeholder 로 대체하지 않는다.
 * - 도움말은 입력 아래, 오류는 그 아래에 원인 + 해결 방법을 함께 적는다.
 * - 오류가 있으면 aria-invalid + aria-describedby 로 스크린리더에도 같이 전달한다.
 */
import { useId } from 'react';
import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes, TextareaHTMLAttributes } from 'react';
import { AlertCircleIcon, ChevronDownIcon } from './Icon';

interface FieldShellProps {
  label: string;
  help?: string;
  /** 원인 + 해결 방법을 한 문장으로. 예: "지급액은 1원 이상이어야 합니다 — 값을 다시 입력하세요." */
  error?: string;
  children: (ids: { controlId: string; describedBy: string | undefined; invalid: boolean }) => ReactNode;
  className?: string;
}

function FieldShell({ label, help, error, children, className }: FieldShellProps) {
  const base = useId();
  const controlId = `${base}-c`;
  const helpId = `${base}-h`;
  const errorId = `${base}-e`;
  const describedBy = [help ? helpId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined;

  return (
    <div className={['field', error ? 'field--invalid' : '', className ?? ''].filter(Boolean).join(' ')}>
      <label className="field__label" htmlFor={controlId}>
        {label}
      </label>
      {children({ controlId, describedBy, invalid: Boolean(error) })}
      {help ? (
        <p className="field__help" id={helpId}>
          {help}
        </p>
      ) : null}
      {error ? (
        <p className="field__error" id={errorId}>
          <AlertCircleIcon size={14} />
          <span>{error}</span>
        </p>
      ) : null}
    </div>
  );
}

export interface TextFieldProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'id' | 'className'> {
  label: string;
  help?: string;
  error?: string;
}

export function TextField({ label, help, error, ...rest }: TextFieldProps) {
  return (
    <FieldShell label={label} help={help} error={error}>
      {({ controlId, describedBy, invalid }) => (
        <input
          id={controlId}
          className="field__control"
          aria-describedby={describedBy}
          aria-invalid={invalid || undefined}
          {...rest}
        />
      )}
    </FieldShell>
  );
}

export interface TextAreaFieldProps
  extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'id' | 'className'> {
  label: string;
  help?: string;
  error?: string;
}

export function TextAreaField({ label, help, error, ...rest }: TextAreaFieldProps) {
  return (
    <FieldShell label={label} help={help} error={error}>
      {({ controlId, describedBy, invalid }) => (
        <textarea
          id={controlId}
          className="field__control"
          aria-describedby={describedBy}
          aria-invalid={invalid || undefined}
          {...rest}
        />
      )}
    </FieldShell>
  );
}

export interface SelectFieldProps
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'id' | 'className'> {
  label: string;
  help?: string;
  error?: string;
  options: Array<{ value: string; label: string }>;
}

export function SelectField({ label, help, error, options, ...rest }: SelectFieldProps) {
  return (
    <FieldShell label={label} help={help} error={error}>
      {({ controlId, describedBy, invalid }) => (
        <span className="field__select">
          <select
            id={controlId}
            className="field__control"
            aria-describedby={describedBy}
            aria-invalid={invalid || undefined}
            {...rest}
          >
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <ChevronDownIcon size={16} />
        </span>
      )}
    </FieldShell>
  );
}
