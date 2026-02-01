import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date): string {
  if (!date) return 'N/A';
  const d = new Date(date);
  if (isNaN(d.getTime())) return 'N/A';
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export function formatDateTime(date: string | Date): string {
  if (!date) return 'N/A';
  const d = new Date(date);
  if (isNaN(d.getTime())) return 'N/A';
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num)
}

export function formatPercent(num: number, decimals = 1): string {
  return `${num.toFixed(decimals)}%`
}

export function getInitials(name: string): string {
  return name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)
}

export function getRiskColor(level: string): string {
  switch (level?.toLowerCase()) {
    case 'critical':
    case 'high':
      return 'text-error-600 bg-error-50'
    case 'medium':
      return 'text-warning-600 bg-warning-50'
    case 'low':
      return 'text-success-600 bg-success-50'
    default:
      return 'text-gray-600 bg-gray-50'
  }
}

export function getStatusColor(status: string): string {
  switch (status?.toLowerCase()) {
    case 'active':
    case 'open':
    case 'deployed':
      return 'text-success-600 bg-success-50'
    case 'pending':
    case 'in_progress':
    case 'pending_approval':
      return 'text-warning-600 bg-warning-50'
    case 'closed':
    case 'resolved':
    case 'retired':
      return 'text-gray-600 bg-gray-50'
    case 'critical':
    case 'rejected':
      return 'text-error-600 bg-error-50'
    default:
      return 'text-primary-600 bg-primary-50'
  }
}

export function getDQIColor(score: number): string {
  if (score >= 90) return 'text-success-600'
  if (score >= 70) return 'text-warning-600'
  return 'text-error-600'
}
