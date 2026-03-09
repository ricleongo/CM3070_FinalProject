/* eslint-disable */
import { FuseNavigationItem } from '@fuse/components/navigation';

export const defaultNavigation: FuseNavigationItem[] = [
    {
        id   : 'dashboard',
        title: 'Dashboard',
        type : 'basic',
        icon : 'heroicons_outline:chart-pie',
        link : '/example'
    },
    {
        id   : 'transfer',
        title: 'Transfer',
        type : 'basic',
        icon : 'heroicons_outline:arrow-right-on-rectangle',
        link : '/example'
    },
    {
        id   : 'wallet',
        title: 'Crypto Wallet',
        type : 'basic',
        icon : 'heroicons_outline:wallet',
        link : '/example'
    },
    {
        id   : 'contributor',
        title: 'Contributor',
        type : 'basic',
        icon : 'heroicons_outline:adjustments-horizontal',
        link : '/example'
    }
];
