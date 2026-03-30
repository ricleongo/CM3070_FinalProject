import { Injectable } from '@angular/core';
import { HttpResponse } from '@angular/common/http';

interface CacheEntry {
  response: HttpResponse<any>;
  entryTime: number;
}

@Injectable({
  providedIn: 'root',
})
export class CacheService {
  private cache = new Map<string, CacheEntry>();
  private readonly CACHE_DURATION = 300000; // 300.000 miliseconds === 5 minutes

  get(key: string): HttpResponse<any> | null {
    const entry = this.cache.get(key);
    if (!entry) {
      return null;
    }

    const isExpired = Date.now() - entry.entryTime > this.CACHE_DURATION;
    if (isExpired) {
      this.cache.delete(key);
      return null;
    }
    return entry.response;
  }

  put(key: string, response: HttpResponse<any>): void {
    this.cache.set(key, { response, entryTime: Date.now() });
  }

  clearCache(): void {
    this.cache.clear();
  }
}
